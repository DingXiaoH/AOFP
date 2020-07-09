# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow benchmark library.

See the README for more information.
"""

from __future__ import print_function


from bc_helpers import *
from bc_base import BenchmarkBase
# from bc_wrapper import TFEvalWrapper, TFTrainWrapper
from tf_utils import log_important, cur_time
from bc_my_supervisor import MySupervisor
from convnet_builder import ConvNetBuilder
from tf_utils import save_hdf5

_DEFAULT_NUM_BATCHES = 100

OVERALL_EVAL_RECORD_FILE = 'bc_overall_eval_records.txt'


class BenchmarkCNN(BenchmarkBase):
    """Class for benchmarking a cnn network."""

    def __init__(self, params, my_params, dataset=None, model=None):
        super(BenchmarkCNN, self).__init__(params=setup(params), my_params=my_params, dataset=dataset, model=model)
        print('running the refactored BC')

    def do_extra_summaries(self, summary_writer, local_step, sess, graph_info):
        pass

    def run(self):
        self.print_info()
        self._prepare_for_compilation()
        #   shawn
        #   You cannot build a bc for train but use it for eval later. You should build another bc for eval and load the weights
        self.graph = tf.Graph()
        if self.params.eval:
            with self.graph.as_default():
                # TODO(laigd): freeze the graph in eval mode.
                (input_producer_op, enqueue_ops, fetches) = self._build_model()

                #   shawn
                if self.my_params.show_variables:
                    self.show_variables()
                self.sess = tf.Session(target='', config=create_config_proto(self.params))

                if self.my_params.just_compile:
                    local_var_init_op = tf.local_variables_initializer()
                    table_init_ops = tf.tables_initializer()
                    variable_mgr_init_ops = [local_var_init_op]
                    if table_init_ops:
                        variable_mgr_init_ops.extend([table_init_ops])
                    with tf.control_dependencies([local_var_init_op]):
                        variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
                    local_var_init_op_group = tf.group(*variable_mgr_init_ops)
                    self.sess.run(local_var_init_op_group)
                    self.input_producer_op = input_producer_op
                    self.enqueue_ops = enqueue_ops
                    self.fetches = fetches
                    print('finished the compilation and initialization process for future evals')
                    self.update_name_to_variables()
                    return

                result = self.eval_loop(input_producer_op, enqueue_ops, fetches)
        else:
            with self.graph.as_default():
                build_result = self._build_graph()
            print('start to preprocess the graph')
            (self.graph, result_to_benchmark) = self._preprocess_graph(self.graph, build_result)
            if self.my_params.show_variables:
                self.show_variables()
            with self.graph.as_default():
                result = self.do_train(result_to_benchmark)

        return result

    def initialize(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        self.last_weights_file = 'NONE'

    #   preconditions:  just_compile=True, run(). This function modify no weights
    def simple_eval(self, eval_record_comment, other_log_file=None, eval_feed_dict=None, save_confusion_matrix_path=None):
        sess = self.sess
        feed_dict = eval_feed_dict or {}

        if self.dataset.queue_runner_required():
            tf.train.start_queue_runners(sess=sess)
        image_producer = None
        if self.input_producer_op is not None:
            image_producer = cnn_util.ImageProducer(
                sess, self.input_producer_op, self.batch_group_size,
                self.params.use_python32_barrier)
            image_producer.start()
        if self.enqueue_ops:
            for i in xrange(len(self.enqueue_ops)):
                sess.run(self.enqueue_ops[:(i + 1)])
                if image_producer is not None:
                    image_producer.notify_image_consumption()
        loop_start_time = start_time = time.time()
        # TODO(laigd): refactor the part to compute/report the accuracy. Currently
        # it only works for image models.
        top_1_accuracy_sum = 0.0
        top_5_accuracy_sum = 0.0
        loss_sum = 0.0
        total_eval_count = self.num_batches * self.batch_size

        if save_confusion_matrix_path is not None:
            confusion_mat = np.zeros((1001, 1001))

        for step in xrange(self.num_batches):
            results = sess.run(self.fetches, feed_dict=feed_dict)
            results = self.model.postprocess(results)
            top_1_accuracy_sum += results.get('top_1_accuracy', 0)
            top_5_accuracy_sum += results.get('top_5_accuracy', 0)
            loss_sum += results['loss']
            if (step + 1) % self.params.display_every == 0:
                duration = time.time() - start_time
                examples_per_sec = (
                    self.batch_size * self.params.display_every / duration)
                log_fn('%i\t%.1f examples/sec' % (step + 1, examples_per_sec))
                start_time = time.time()
            if image_producer is not None:
                image_producer.notify_image_consumption()
            if save_confusion_matrix_path is not None:
                predicted_labels = np.argmax(results['logits'], axis=1)
                groundtruth_labels = results['labels']
                print('predicted_labels:', predicted_labels)
                print('groundtruth_labels:', groundtruth_labels)
                for k in range(len(predicted_labels)):
                    confusion_mat[groundtruth_labels[k],predicted_labels[k]] += 1

        loop_end_time = time.time()
        if image_producer is not None:
            image_producer.done()
        accuracy_at_1 = top_1_accuracy_sum / self.num_batches
        accuracy_at_5 = top_5_accuracy_sum / self.num_batches
        mean_loss = loss_sum / self.num_batches
        summary = tf.Summary()
        summary.value.add(tag='eval/Accuracy@1', simple_value=accuracy_at_1)
        summary.value.add(tag='eval/Accuracy@5', simple_value=accuracy_at_5)
        for result_key, result_value in results.items():
            if result_key.startswith(constants.SIMPLE_VALUE_RESULT_PREFIX):
                prefix_len = len(constants.SIMPLE_VALUE_RESULT_PREFIX)
                summary.value.add(tag='eval/' + result_key[prefix_len:],
                    simple_value=result_value)

        log_fn('Accuracy @ 1 = %.4f Accuracy @ 5 = %.4f Loss = %.8f [%d examples]' %
               (accuracy_at_1, accuracy_at_5, mean_loss, total_eval_count))
        elapsed_time = loop_end_time - loop_start_time
        images_per_sec = (self.num_batches * self.batch_size / elapsed_time)
        # Note that we compute the top 1 accuracy and top 5 accuracy for each
        # batch, which will have a slight performance impact.

        log_fn('-' * 64)
        log_fn('total images/sec: %.2f' % images_per_sec)
        log_fn('-' * 64)

        lf = other_log_file or self.my_params.eval_log_file or OVERALL_EVAL_RECORD_FILE
        log_important('{},{},top1={:.5f},top5={:.5f},loss={:.9f} on {} at {}, {}'.format(self.params.model, self.last_weights_file or self.my_params.load_ckpt or self.my_params.init_hdf5,
            accuracy_at_1, accuracy_at_5, mean_loss, self.subset, cur_time(), eval_record_comment), log_file=lf)

        if save_confusion_matrix_path is not None:
            np.save(save_confusion_matrix_path, {'conf_mat': confusion_mat})

        return {'top1':accuracy_at_1, 'top5':accuracy_at_5, 'mean_loss':mean_loss}


    def simple_survey(self, survey_save_file, survey_layer_ids, bias_opt, agg_opt, eval_feed_dict=None):
        assert bias_opt in ['none', 'biased', 'both']
        assert agg_opt in ['apop', 'gap', 'taylor']   #TODO how to implement taylor-based?
        assert self.my_params.need_record_internal_outputs
        sess = self.sess
        feed_dict = eval_feed_dict or {}

        if self.dataset.queue_runner_required():
            tf.train.start_queue_runners(sess=sess)
        image_producer = None
        if self.input_producer_op is not None:
            image_producer = cnn_util.ImageProducer(
                sess, self.input_producer_op, self.batch_group_size,
                self.params.use_python32_barrier)
            image_producer.start()
        if self.enqueue_ops:
            for i in xrange(len(self.enqueue_ops)):
                sess.run(self.enqueue_ops[:(i + 1)])
                if image_producer is not None:
                    image_producer.notify_image_consumption()
        loop_start_time = start_time = time.time()

        internal_outputs_dict = self.fetches['internal_outputs']

        aggregated_outputs_dict = {}

        if agg_opt in ['gap', 'apop']:
            for name, out in internal_outputs_dict.items():
                if bias_opt == 'biased':
                    if '#' not in name:
                        continue
                    layer_idx = int(name.split('#')[0])
                    if layer_idx not in survey_layer_ids:
                        continue
                else:
                    assert False
                axes = [1, 2] if self.params.data_format == 'NHWC' else [2, 3]
                if agg_opt == 'gap':
                    aggregated_outputs_dict[layer_idx] = tf.reduce_mean(out, axes, keepdims=False)
                else:
                    assert self.params.data_format == 'NHWC'
                    aggregated_outputs_dict[layer_idx] = tf.reduce_sum(tf.cast(out > 0, tf.float32),
                        axis=[1, 2]) / tf.cast(tf.shape(out)[1] * tf.shape(out)[2], tf.float32)
        else:
            assert agg_opt == 'taylor'
            assert len(survey_layer_ids) == 1
            # TODO actually 'internal_outputs_dict' here is not used. we use this framework just for convenient
            kernel_to_survey = self.get_kernel_variables()[survey_layer_ids[0]]
            print('the kernel to taylor-survey is ', kernel_to_survey.name)
            optimizer = tf.train.MomentumOptimizer(learning_rate=0, momentum=0.9)
            grads_and_vars = optimizer.compute_gradients(loss=self.fetches['loss'], var_list=[kernel_to_survey])
            assert len(grads_and_vars) == 1
            print('grads_and_vars: ', grads_and_vars)
            grad_to_survey = grads_and_vars[0][0]
            aggregated_outputs_dict[survey_layer_ids[0]] = tf.expand_dims(tf.reduce_sum(tf.abs(grad_to_survey * kernel_to_survey), axis=[0,1,2], keepdims=False), axis=0)



        output_record_dict = {idx: list() for idx in aggregated_outputs_dict.keys()}

        total_eval_count = self.num_batches * self.batch_size

        for step in xrange(self.num_batches):
            results = sess.run(aggregated_outputs_dict, feed_dict=feed_dict)
            for out_idx, out_array in results.items():
                output_record_dict[out_idx].append(out_array)
            # results = self.model.postprocess(results)
            if (step + 1) % self.params.display_every == 0:
                duration = time.time() - start_time
                examples_per_sec = (
                    self.batch_size * self.params.display_every / duration)
                log_fn('%i\t%.1f examples/sec' % (step + 1, examples_per_sec))
                start_time = time.time()
            if image_producer is not None:
                image_producer.notify_image_consumption()
        loop_end_time = time.time()
        if image_producer is not None:
            image_producer.done()

        elapsed_time = loop_end_time - loop_start_time
        images_per_sec = (self.num_batches * self.batch_size / elapsed_time)
        log_fn('-' * 64)
        log_fn('total images/sec: %.2f' % images_per_sec)
        log_fn('-' * 64)

        save_dict = {}
        for idx, array_list in output_record_dict.items():
            save_dict[idx] = np.concatenate(array_list, axis=0)
            assert save_dict[idx].shape[0] == total_eval_count

        print('save the survey hdf5 to {}, where the keys are {}'.format(survey_save_file, save_dict.keys()))
        save_hdf5(save_dict, survey_save_file)
        return save_dict



    def _prepare_for_compilation(self):
        """Run the benchmark task assigned to this process.

        Returns:
          Dictionary of statistics for training or eval.
        Raises:
           ValueError: unrecognized job name.
        """
        if self.params.job_name == 'ps':
            log_fn('Running parameter server %s' % self.task_index)
            self.cluster_manager.join_server()
            return {}

        # For distributed_all_reduce with multiple workers, drive
        # from a separate controller process.
        if self.params.variable_update == 'distributed_all_reduce':
            if self.params.job_name == 'worker':
                log_fn('Starting worker %s' % self.task_index)
                self.cluster_manager.join_server()
                return
            elif self.params.job_name and self.params.job_name != 'controller':
                raise ValueError('unrecognized job name: %s' % self.params.job_name)

        self._log_benchmark_run()

    #
    # #   return a wrapper for eval, the wrapper should have (input_producer_op, enqueue_ops, fetches)
    # def compile_for_eval(self):
    #     assert self.params.eval
    #     self.graph = tf.Graph()
    #     with self.graph.as_default():
    #         # TODO(laigd): freeze the graph in eval mode.
    #         (input_producer_op, enqueue_ops, fetches) = self._build_model()
    #     return TFEvalWrapper(self.graph, input_producer_op=input_producer_op, enqueue_ops=enqueue_ops, fetches=fetches)
    #
    #
    # #   return a wrapper for train
    # def compile_for_train(self):
    #     assert not self.params.eval
    #     self.graph = tf.Graph()
    #     with self.graph.as_default():
    #         build_result = self._build_graph()
    #     (self.graph, result_to_benchmark) = self._preprocess_graph(self.graph, build_result)
    #     return TFTrainWrapper(self.graph, graph_info=result_to_benchmark)
    #




    def eval_loop(self, input_producer_op, enqueue_ops, fetches):
        """Evaluate a model every self.params.eval_interval_secs.

        Returns:
          Dictionary containing eval statistics. Currently returns an empty
          dictionary.
        """
        saver = tf.train.Saver(self.variable_mgr.savable_variables())
        if self.params.eval_dir is not None:
            summary_writer = tf.summary.FileWriter(self.params.eval_dir,
                tf.get_default_graph())
            summary_op = tf.summary.merge_all()
        else:
            summary_writer = None
            summary_op = None
        target = ''
        local_var_init_op = tf.local_variables_initializer()
        table_init_ops = tf.tables_initializer()
        variable_mgr_init_ops = [local_var_init_op]
        if table_init_ops:
            variable_mgr_init_ops.extend([table_init_ops])
        with tf.control_dependencies([local_var_init_op]):
            variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
        local_var_init_op_group = tf.group(*variable_mgr_init_ops)

        # TODO(huangyp): Check if checkpoints haven't updated for hours and abort.
        while True:
            self.eval_once(saver, summary_writer, target, local_var_init_op_group,
                input_producer_op, enqueue_ops, fetches, summary_op)
            if self.params.eval_interval_secs <= 0:
                break
            time.sleep(self.params.eval_interval_secs)
        return {}



    def eval_once(self, saver, summary_writer, target, local_var_init_op_group,
                  input_producer_op, enqueue_ops, fetches, summary_op, eval_feed_dict=None):
        """Evaluate the model from a checkpoint using validation dataset."""

        sess = self.sess

        sess.run(local_var_init_op_group)

        feed_dict = eval_feed_dict or {}


        if self.my_params.init_hdf5:
            print('got the init_hdf5 param, so start loading weights from it')
            assert self.params.eval_dir is None
            self.load_weights_from_hdf5(self.my_params.init_hdf5)
            global_step = tf.constant(0, dtype=tf.int32)
        else:
            if self.params.train_dir is None:
                print('note that train_dir is not specified')
                # raise ValueError('Trained model directory not specified')
            try:
                global_step, checkpoint_path = load_checkpoint(saver, sess, self.my_params.load_ckpt)
                self.last_weights_file = checkpoint_path
                print('got global_step={} in checkpoint {}'.format(global_step, checkpoint_path))
            except CheckpointNotFoundException:
                log_fn('Checkpoint not found in %s' % self.my_params.load_ckpt)
                return

        if self.dataset.queue_runner_required():
            tf.train.start_queue_runners(sess=sess)
        image_producer = None
        if input_producer_op is not None:
            image_producer = cnn_util.ImageProducer(
                sess, input_producer_op, self.batch_group_size,
                self.params.use_python32_barrier)
            image_producer.start()
        if enqueue_ops:
            for i in xrange(len(enqueue_ops)):
                sess.run(enqueue_ops[:(i + 1)])
                if image_producer is not None:
                    image_producer.notify_image_consumption()
        loop_start_time = start_time = time.time()
        # TODO(laigd): refactor the part to compute/report the accuracy. Currently
        # it only works for image models.
        top_1_accuracy_sum = 0.0
        top_5_accuracy_sum = 0.0
        loss_sum = 0.0
        total_eval_count = self.num_batches * self.batch_size
        print('total_eval_count=', total_eval_count)

        # print('----------show var values before eval---------')
        # for v in self.get_global_variables():
        #     if 'global' in v.name:
        #         continue
        #     print(v.name, np.mean(sess.run(v)))

        for step in xrange(self.num_batches):
            if (summary_writer is not None and summary_op is not None and self.params.save_summaries_steps > 0 and
                            (step + 1) % self.params.save_summaries_steps == 0):
                results, summary_str = sess.run([fetches, summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary_str)
            else:
                results = sess.run(fetches, feed_dict=feed_dict)
            results = self.model.postprocess(results)
            top_1_accuracy_sum += results['top_1_accuracy']
            top_5_accuracy_sum += results['top_5_accuracy']
            loss_sum += results['loss']
            if (step + 1) % self.params.display_every == 0:
                duration = time.time() - start_time
                examples_per_sec = (
                    self.batch_size * self.params.display_every / duration)
                log_fn('%i\t%.1f examples/sec' % (step + 1, examples_per_sec))
                start_time = time.time()
            if image_producer is not None:
                image_producer.notify_image_consumption()
        loop_end_time = time.time()
        if image_producer is not None:
            image_producer.done()
        accuracy_at_1 = top_1_accuracy_sum / self.num_batches
        print('top1={}/{}={}'.format(top_1_accuracy_sum, self.num_batches, accuracy_at_1))
        accuracy_at_5 = top_5_accuracy_sum / self.num_batches
        mean_loss = loss_sum / self.num_batches
        summary = tf.Summary()
        summary.value.add(tag='eval/Accuracy@1', simple_value=accuracy_at_1)
        summary.value.add(tag='eval/Accuracy@5', simple_value=accuracy_at_5)
        for result_key, result_value in results.items():
            if result_key.startswith(constants.SIMPLE_VALUE_RESULT_PREFIX):
                prefix_len = len(constants.SIMPLE_VALUE_RESULT_PREFIX)
                summary.value.add(tag='eval/' + result_key[prefix_len:],
                    simple_value=result_value)
        if summary_writer is not None:
            summary_writer.add_summary(summary, global_step)
        log_fn('Accuracy @ 1 = %.4f Accuracy @ 5 = %.4f Loss = %.8f [%d examples]' %
               (accuracy_at_1, accuracy_at_5, mean_loss, total_eval_count))
        elapsed_time = loop_end_time - loop_start_time
        images_per_sec = (self.num_batches * self.batch_size / elapsed_time)
        # Note that we compute the top 1 accuracy and top 5 accuracy for each
        # batch, which will have a slight performance impact.

        if self.my_params.save_hdf5:
            self.save_weights_to_hdf5(self.my_params.save_hdf5)

        log_fn('-' * 64)
        log_fn('total images/sec: %.2f' % images_per_sec)
        log_fn('-' * 64)
        if self.benchmark_logger:
            eval_result = {
                'eval_top_1_accuracy', accuracy_at_1,
                'eval_top_5_accuracy', accuracy_at_5,
                'eval_average_examples_per_sec', images_per_sec,
                tf.GraphKeys.GLOBAL_STEP, global_step,
            }
            self.benchmark_logger.log_evaluation_result(eval_result)

        lf = self.my_params.eval_log_file or OVERALL_EVAL_RECORD_FILE
        log_important('{},{},top1={:.5f},top5={:.5f},loss={:.8f} on {} at {}'.format(self.params.model, self.last_weights_file or self.my_params.load_ckpt or self.my_params.init_hdf5,
            accuracy_at_1, accuracy_at_5, mean_loss, self.subset, cur_time()), log_file=lf)






    GPU_CACHED_INPUT_VARIABLE_NAME = 'gpu_cached_inputs'


    #   shawn
    #   overwrite this if you wish to use another convnet_builder (bds convnet builder, for example)
    def get_convnet_builder(self, input_list, phase_train):
        images = input_list[0]
        assert self.params.data_format in ['NCHW', 'NHWC']
        if self.params.data_format == 'NCHW':
            images = tf.transpose(images, [0, 3, 1, 2])
        var_type = tf.float32
        data_type = tf.float16 if self.params.use_fp16 else tf.float32
        if data_type == tf.float16 and self.params.fp16_vars:
            var_type = tf.float16
        #   shawn,
        #   input_nchan=3 any exceptions ?
        convnet_builder = ConvNetBuilder(images, input_nchan=3, phase_train=phase_train, use_tf_layers=self.params.use_tf_layers,
            data_format=self.params.data_format, dtype=data_type, variable_dtype=var_type, use_dense_layer=self.my_params.use_dense_layer, input_rotation=self.my_params.input_rotation)
        return convnet_builder


    def postprocess_after_build_by_convnet_builder(self, convnet_builder, build_results):
        print('nothing to do after build by convnet builder')


    def do_train(self, graph_info):
        """Benchmark the graph.

        Args:
          graph_info: the namedtuple returned by _build_graph() which
            contains all necessary information to benchmark the graph, including
            named tensors/ops list, fetches, etc.
        Returns:
          Dictionary containing training statistics (num_workers, num_steps,
          average_wall_time, images_per_sec).
        """
        if self.params.variable_update == 'horovod':
            import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
            # First worker will be 'chief' - it will write summaries and
            # save checkpoints.
            is_chief = hvd.rank() == 0
        else:
            is_chief = (not self.job_name or self.task_index == 0)

        summary_op = tf.summary.merge_all()
        # summary_op = tf.group(summary_op, graph_info.summary_op_group)
        # summary_op = tf.group(*graph_info.summary_ops)

        summary_writer = None
        if (is_chief and self.params.summary_verbosity and self.params.train_dir and
                    self.params.save_summaries_steps > 0):
            summary_writer = tf.summary.FileWriter(self.params.train_dir,
                tf.get_default_graph())

        # We want to start the benchmark timer right after a image_producer barrier
        # and avoids undesired waiting times on barriers.
        if ((self.num_warmup_batches + len(graph_info.enqueue_ops) - 1) %
                self.batch_group_size) != 0:
            self.num_warmup_batches = int(
                math.ceil(
                    (self.num_warmup_batches + len(graph_info.enqueue_ops) - 1.0) /
                    (self.batch_group_size)) * self.batch_group_size -
                len(graph_info.enqueue_ops) + 1)
            log_fn('Round up warm up steps to %d to match batch_group_size' %
                   self.num_warmup_batches)
            assert ((self.num_warmup_batches + len(graph_info.enqueue_ops) - 1) %
                    self.batch_group_size) == 0
        # We run the summaries in the same thread as the training operations by
        # passing in None for summary_op to avoid a summary_thread being started.
        # Running summaries and training operations in parallel could run out of
        # GPU memory.
        if is_chief and not self.forward_only_and_freeze:
            saver = tf.train.Saver(
                self.variable_mgr.savable_variables(),
                save_relative_paths=True,
                max_to_keep=self.params.max_ckpts_to_keep)
        else:
            saver = None
        ready_for_local_init_op = None
        if self.job_name and not (self.single_session or
                                      self.distributed_collective):
            # In distributed mode, we don't want to run local_var_init_op_group until
            # the global variables are initialized, because local_var_init_op_group
            # may use global variables (such as in distributed replicated mode). We
            # don't set this in non-distributed mode, because in non-distributed mode,
            # local_var_init_op_group may itself initialize global variables (such as
            # in replicated mode).
            ready_for_local_init_op = tf.report_uninitialized_variables(
                tf.global_variables())
        if self.params.variable_update == 'horovod':
            import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
            bcast_global_variables_op = hvd.broadcast_global_variables(0)
        else:
            bcast_global_variables_op = None

        if self.params.variable_update == 'collective_all_reduce':
            # It doesn't matter what this collective_graph_key value is,
            # so long as it's > 0 and the same at every worker.
            init_run_options = tf.RunOptions()
            init_run_options.experimental.collective_graph_key = 6
        else:
            init_run_options = tf.RunOptions()
        sv = MySupervisor(
            # For the purpose of Supervisor, all Horovod workers are 'chiefs',
            # since we want session to be initialized symmetrically on all the
            # workers.
            is_chief=is_chief or (self.params.variable_update == 'horovod'
                                  or self.distributed_collective),
            # Log dir should be unset on non-chief workers to prevent Horovod
            # workers from corrupting each other's checkpoints.
            logdir=self.params.train_dir if is_chief else None,
            ready_for_local_init_op=ready_for_local_init_op,
            local_init_op=graph_info.local_var_init_op_group,
            saver=saver,
            global_step=graph_info.global_step,
            summary_op=None,
            save_model_secs=self.params.save_model_secs,
            summary_writer=summary_writer,
            local_init_run_options=init_run_options,
            load_ckpt_full_path=self.my_params.load_ckpt,
            auto_continue=self.my_params.auto_continue)



        step_train_times = []
        start_standard_services = (
            self.params.train_dir or
            self.dataset.queue_runner_required())
        target = self.cluster_manager.get_target() if self.cluster_manager else ''

        #shawn
        sess_context = sv.managed_session(
                master=target,
                config=create_config_proto(self.params),
                start_standard_services=start_standard_services)

        with sess_context as sess:

            self.sess = sess

            if self.params.backbone_model_path is not None:
                self.model.load_backbone_model(sess, self.params.backbone_model_path)
            if bcast_global_variables_op:
                sess.run(bcast_global_variables_op)

            image_producer = None
            if graph_info.input_producer_op is not None:
                image_producer = cnn_util.ImageProducer(
                    sess, graph_info.input_producer_op, self.batch_group_size,
                    self.params.use_python32_barrier)
                image_producer.start()
            if graph_info.enqueue_ops:
                for i in xrange(len(graph_info.enqueue_ops)):
                    sess.run(graph_info.enqueue_ops[:(i + 1)])
                    if image_producer is not None:
                        image_producer.notify_image_consumption()
            self.init_global_step, = sess.run([graph_info.global_step])
            print('the current global step is ', self.init_global_step)
            if self.job_name and not self.params.cross_replica_sync:
                # TODO(zhengxq): Do we need to use a global step watcher at all?
                global_step_watcher = GlobalStepWatcher(
                    sess, graph_info.global_step,
                    self.num_workers * self.num_warmup_batches +
                    self.init_global_step,
                    self.num_workers * (self.num_warmup_batches + self.num_batches) - 1)
                global_step_watcher.start()
            else:
                global_step_watcher = None

            if self.graph_file is not None:
                path, filename = os.path.split(self.graph_file)
                as_text = filename.endswith('txt')
                log_fn('Writing GraphDef as %s to %s' % (  # pyformat break
                    'text' if as_text else 'binary', self.graph_file))
                tf.train.write_graph(sess.graph.as_graph_def(add_shapes=True), path,
                    filename, as_text)

            log_fn('Running warm up')
            local_step = -1 * self.num_warmup_batches
            if self.single_session:
                # In single session mode, each step, the global_step is incremented by
                # 1. In non-single session mode, each step, the global_step is
                # incremented once per worker. This means we need to divide
                # init_global_step by num_workers only in non-single session mode.
                end_local_step = self.num_batches - self.init_global_step
            else:
                end_local_step = self.num_batches - (self.init_global_step /
                                                     self.num_workers)

            if not global_step_watcher:
                # In cross-replica sync mode, all workers must run the same number of
                # local steps, or else the workers running the extra step will block.
                done_fn = lambda: local_step >= end_local_step
            else:
                done_fn = global_step_watcher.done
            if self.params.debugger is not None:
                if self.params.debugger == 'cli':
                    log_fn('The CLI TensorFlow debugger will be used.')
                    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                    self.sess = sess
                else:
                    log_fn('The TensorBoard debugger plugin will be used.')
                    sess = tf_debug.TensorBoardDebugWrapperSession(sess, self.params.debugger)
                    self.sess = sess
            profiler = tf.profiler.Profiler() if self.params.tfprof_file else None
            loop_start_time = time.time()
            last_average_loss = None

            ##########
            #   shawn
            # if self.my_params.init_hdf5:
            #     self.load_weights_from_hdf5(self.my_params.init_hdf5)

            # print('----------show var values before train---------')
            # for v in self.get_global_variables():
            #     if 'global' in v.name:
            #         continue
            #     print(v.name, np.mean(sess.run(v)))


            print('self.lr_boundaries=', self.lr_boundaries)


            while not done_fn():
                if local_step == 0:
                    log_fn('Done warm up')
                    if graph_info.execution_barrier:
                        log_fn('Waiting for other replicas to finish warm up')
                        sess.run([graph_info.execution_barrier])

                    # TODO(laigd): rename 'Img' to maybe 'Input'.
                    header_str = ('Step\tImg/sec\t' +
                                  self.params.loss_type_to_report.replace('/', ' '))
                    if self.params.print_training_accuracy or self.params.forward_only:
                        # TODO(laigd): use the actual accuracy op names of the model.
                        header_str += '\ttop_1_accuracy\ttop_5_accuracy'
                    log_fn(header_str)
                    assert len(step_train_times) == self.num_warmup_batches
                    # reset times to ignore warm up batch
                    step_train_times = []
                    loop_start_time = time.time()
                if (summary_writer and (local_step + 1) % self.params.save_summaries_steps == 0):
                    fetch_summary = summary_op
                else:
                    fetch_summary = None
                collective_graph_key = 7 if (self.params.variable_update == 'collective_all_reduce') else 0


                (summary_str, last_average_loss, _) = benchmark_one_step(
                    sess, graph_info.fetches, local_step,
                    self.batch_size * (self.num_workers
                                       if self.single_session else 1), step_train_times,
                    self.trace_filename, self.params.partitioned_graph_file_prefix,
                    profiler, image_producer, self.params, fetch_summary,
                    benchmark_logger=self.benchmark_logger,
                    collective_graph_key=collective_graph_key,
                    track_mvav_op=graph_info.mvav_op)

                local_step += 1

                if summary_str is not None and is_chief:
                    sv.summary_computed(sess, summary_str)

                self.do_extra_summaries(summary_writer = summary_writer, local_step = local_step, sess=sess, graph_info=graph_info)

                if (self.my_params.num_steps_per_hdf5 > 0 and local_step % self.my_params.num_steps_per_hdf5 == 0 and local_step > 0 and is_chief):
                    self.save_hdf5_by_global_step(sess.run(graph_info.global_step))

                if (self.params.save_model_steps and local_step % self.params.save_model_steps == 0 and local_step > 0 and is_chief):
                    sv.saver.save(sess, sv.save_path, sv.global_step)

                if self.lr_boundaries is not None and local_step % 100 == 0 and local_step > 0 and is_chief:
                    cur_global_step = sess.run(graph_info.global_step)
                    for b in self.lr_boundaries:
                        if b > cur_global_step and b - cur_global_step < 100:
                            sv.saver.save(sess, sv.save_path, sv.global_step)
                            self.save_hdf5_by_global_step(cur_global_step)
                            break

                if self.my_params.frequently_save_interval is not None and self.my_params.frequently_save_last_epochs is not None and local_step % self.my_params.frequently_save_interval == 0 and local_step > 0 and is_chief:
                    cur_global_step = sess.run(graph_info.global_step)
                    remain_steps = self.num_batches - cur_global_step
                    remain_epochs = remain_steps * self.batch_size / self.dataset.num_examples_per_epoch(self.subset)
                    if remain_epochs < self.my_params.frequently_save_last_epochs:
                        self.save_hdf5_by_global_step(cur_global_step)






            loop_end_time = time.time()
            # Waits for the global step to be done, regardless of done_fn.
            if global_step_watcher:
                while not global_step_watcher.done():
                    time.sleep(.25)
            if not global_step_watcher:
                elapsed_time = loop_end_time - loop_start_time
                average_wall_time = elapsed_time / local_step if local_step > 0 else 0
                images_per_sec = (self.num_workers * local_step * self.batch_size /
                                  elapsed_time)
                num_steps = local_step * self.num_workers
            else:
                # NOTE: Each worker independently increases the global step. So,
                # num_steps will be the sum of the local_steps from each worker.
                num_steps = global_step_watcher.num_steps()
                elapsed_time = global_step_watcher.elapsed_time()
                average_wall_time = (elapsed_time * self.num_workers / num_steps
                                     if num_steps > 0 else 0)
                images_per_sec = num_steps * self.batch_size / elapsed_time

            if self.my_params.save_hdf5:
                print('start saving the final hdf5 to ', self.my_params.save_hdf5)
                self.save_weights_to_hdf5(self.my_params.save_hdf5)
                if self.my_params.save_mvav:
                    self.save_moving_average_weights_to_hdf5(self.my_params.save_hdf5.replace('.hdf5', '_mvav.hdf5'),
                        moving_averages=self.variable_averages)

            log_fn('-' * 64)
            # TODO(laigd): rename 'images' to maybe 'inputs'.
            log_fn('total images/sec: %.2f' % images_per_sec)
            log_fn('-' * 64)
            if image_producer is not None:
                image_producer.done()
            if is_chief:
                if self.benchmark_logger:
                    self.benchmark_logger.log_metric(
                        'average_examples_per_sec', images_per_sec, global_step=num_steps)

            # Save the model checkpoint.
            if self.params.train_dir is not None and is_chief:
                checkpoint_path = os.path.join(self.params.train_dir, 'model.ckpt')
                if not gfile.Exists(self.params.train_dir):
                    gfile.MakeDirs(self.params.train_dir)
                sv.saver.save(sess, checkpoint_path, graph_info.global_step)

            if graph_info.execution_barrier:
                # Wait for other workers to reach the end, so this worker doesn't
                # go away underneath them.
                sess.run([graph_info.execution_barrier])


        sv.stop()
        if profiler:
            generate_tfprof_profile(profiler, self.params.tfprof_file)
        stats = {
            'num_workers': self.num_workers,
            'num_steps': num_steps,
            'average_wall_time': average_wall_time,
            'images_per_sec': images_per_sec
        }
        if last_average_loss is not None:
            stats['last_average_loss'] = last_average_loss
        return stats

    def save_hdf5_by_global_step(self, cur_global_step):
        save_path = os.path.join(self.params.train_dir, 'ckpt_step_{}.hdf5'.format(cur_global_step))
        print('saving hdf5 to ', save_path)
        self.save_weights_to_hdf5(hdf5_file=save_path)
        if self.my_params.save_mvav:
            self.save_moving_average_weights_to_hdf5(save_path.replace('.hdf5', '_mvav.hdf5'),
                moving_averages=self.variable_averages)


    def add_forward_pass_and_gradients(self,
                                       phase_train,
                                       rel_device_num,
                                       abs_device_num,
                                       input_processing_info,
                                       gpu_compute_stage_ops,
                                       gpu_grad_stage_ops):
        """Add ops for forward-pass and gradient computations."""
        nclass = self.dataset.num_classes
        if self.datasets_use_prefetch:
            function_buffering_resource = None
            if input_processing_info.function_buffering_resources:
                function_buffering_resource = (
                    input_processing_info.function_buffering_resources[rel_device_num])

            input_data = None
            if input_processing_info.multi_device_iterator_input:
                input_data = (
                    input_processing_info.multi_device_iterator_input[rel_device_num])

            # Exactly one of function_buffering_resource or input_data is not None.
            if function_buffering_resource is None and input_data is None:
                raise ValueError('Both function_buffering_resource and input_data '
                                 'cannot be null if datasets_use_prefetch=True')
            if function_buffering_resource is not None and input_data is not None:
                raise ValueError('Both function_buffering_resource and input_data '
                                 'cannot be specified. Only one should be.')
            with tf.device(self.raw_devices[rel_device_num]):
                if function_buffering_resource is not None:
                    input_list = prefetching_ops.function_buffering_resource_get_next(
                        function_buffering_resource,
                        output_types=self.model.get_input_data_types())
                else:
                    input_list = input_data
        else:
            if not self.dataset.use_synthetic_gpu_inputs():
                input_producer_stage = input_processing_info.input_producer_stages[
                    rel_device_num]
                with tf.device(self.cpu_device):
                    host_input_list = input_producer_stage.get()
                with tf.device(self.raw_devices[rel_device_num]):
                    gpu_compute_stage = data_flow_ops.StagingArea(
                        [inp.dtype for inp in host_input_list],
                        shapes=[inp.get_shape() for inp in host_input_list])
                    # The CPU-to-GPU copy is triggered here.
                    gpu_compute_stage_op = gpu_compute_stage.put(host_input_list)
                    input_list = gpu_compute_stage.get()
                    gpu_compute_stage_ops.append(gpu_compute_stage_op)
            else:
                with tf.device(self.raw_devices[rel_device_num]):
                    # Minor hack to avoid H2D copy when using synthetic data
                    input_list = self.model.get_synthetic_inputs(
                        BenchmarkCNN.GPU_CACHED_INPUT_VARIABLE_NAME, nclass)

        with tf.device(self.devices[rel_device_num]):
            input_shapes = self.model.get_input_shapes()
            input_list = [
                tf.reshape(input_list[i], shape=input_shapes[i])
                for i in range(len(input_list))
            ]

        def forward_pass_and_gradients():
            """Builds forward pass and gradient computation network.

            When phase_train=True and print_training_accuracy=False:
              return [loss] + grads

            When phase_train=True and print_training_accuracy=True:
              return [logits, loss] + grads

            When phase_train=False,
              return [logits]

            Its output can always be unpacked by

            ```
              outputs = forward_pass_and_gradients()
              logits, loss, grads = unpack_forward_pass_and_gradients_output(outputs)
            ```

            Returns:
              outputs: A list of tensors depending on different modes.
            """

            self.convnet_builder = self.get_convnet_builder(input_list=input_list, phase_train=phase_train)
            if self.my_params.need_record_internal_outputs:
                self.convnet_builder.enable_record_internal_outputs()

            build_network_result = self.model.build_network(
                self.convnet_builder, nclass)
            logits = build_network_result.logits

            base_loss = self.model.loss_function(input_list, build_network_result)

            self.postprocess_after_build_by_convnet_builder(self.convnet_builder, build_network_result)

            if not phase_train:
                assert self.num_gpus == 1
                eval_fetches = [logits, base_loss]
                return eval_fetches

            params = self.variable_mgr.trainable_variables_on_device(
                rel_device_num, abs_device_num)

            l2_loss = None
            total_loss = base_loss
            with tf.name_scope('l2_loss'):
                if self.my_params.apply_l2_on_vector_params:
                    params_to_regularize = [p for p in params]
                else:
                    params_to_regularize = [p for p in params if len(p.get_shape()) in [2, 4]]
                print('add l2 loss on these params:', [p.name for p in params_to_regularize])
                if self.model.data_type == tf.float16 and self.params.fp16_vars:
                    # fp16 reductions are very slow on GPUs, so cast to fp32 before
                    # calling tf.nn.l2_loss and tf.add_n.
                    # TODO(b/36217816): Once the bug is fixed, investigate if we should do
                    # this reduction in fp16.
                    params_to_regularize = (tf.cast(p, tf.float32) for p in params_to_regularize)
                if rel_device_num == len(self.devices) - 1:
                    # We compute the L2 loss for only one device instead of all of them,
                    # because the L2 loss for each device is the same. To adjust for this,
                    # we multiply the L2 loss by the number of devices. We choose the
                    # last device because for some reason, on a Volta DGX1, the first four
                    # GPUs take slightly longer to complete a step than the last four.
                    # TODO(reedwm): Shard the L2 loss computations across GPUs.
                    custom_l2_loss = self.model.custom_l2_loss(params_to_regularize)
                    if custom_l2_loss is not None:
                        l2_loss = custom_l2_loss
                        print('use the custom l2 loss')
                    elif self.params.single_l2_loss_op:
                        # TODO(reedwm): If faster, create a fused op that does the L2 loss
                        # on multiple tensors, and use that instead of concatenating
                        # tensors.
                        reshaped_params = [tf.reshape(p, (-1,)) for p in params_to_regularize]
                        l2_loss = tf.nn.l2_loss(tf.concat(reshaped_params, axis=0))
                    else:
                        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params_to_regularize])
            weight_decay = self.params.weight_decay
            if (weight_decay is not None and weight_decay != 0. and
                        l2_loss is not None):
                print('the l2 loss factor (weight decay) is ', weight_decay)
                total_loss += len(self.devices) * weight_decay * l2_loss

            aggmeth = tf.AggregationMethod.DEFAULT
            scaled_loss = (total_loss if self.loss_scale is None
                           else total_loss * self.loss_scale)
            grads = tf.gradients(scaled_loss, params, aggregation_method=aggmeth)
            if self.loss_scale is not None:
                # TODO(reedwm): If automatic loss scaling is not used, we could avoid
                # these multiplications by directly modifying the learning rate instead.
                # If this is done, care must be taken to ensure that this scaling method
                # is correct, as some optimizers square gradients and do other
                # operations which might not be compatible with modifying both the
                # gradients and the learning rate.

                grads = [
                    grad * tf.cast(1. / self.loss_scale, grad.dtype) for grad in grads
                ]

            if self.params.variable_update == 'horovod':
                import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
                if self.params.horovod_device:
                    horovod_device = '/%s:0' % self.params.horovod_device
                else:
                    horovod_device = ''
                # All-reduce gradients using Horovod.
                grads = [hvd.allreduce(grad, average=False, device_dense=horovod_device)
                         for grad in grads]

            if self.params.staged_vars:
                grad_dtypes = [grad.dtype for grad in grads]
                grad_shapes = [grad.shape for grad in grads]
                grad_stage = data_flow_ops.StagingArea(grad_dtypes, grad_shapes)
                grad_stage_op = grad_stage.put(grads)
                # In general, this decouples the computation of the gradients and
                # the updates of the weights.
                # During the pipeline warm up, this runs enough training to produce
                # the first set of gradients.
                gpu_grad_stage_ops.append(grad_stage_op)
                grads = grad_stage.get()

            if self.params.loss_type_to_report == 'total_loss':
                loss = total_loss
            else:
                loss = base_loss

            if self.params.print_training_accuracy:
                return [logits, loss] + grads
            else:
                return [loss] + grads

        def unpack_forward_pass_and_gradients_output(forward_pass_and_grad_outputs):
            """Unpacks outputs from forward_pass_and_gradients.

            Args:
              forward_pass_and_grad_outputs: Output from forward_pass_and_gradients.

            Returns:
              logits: Unscaled probability distribution from forward pass.
                If unavailable, None is returned.
              loss: Loss function result from logits.
                If unavailable, None is returned.
              grads: Gradients for all trainable variables.
                If unavailable, None is returned.
            """
            logits = None
            # logits is only fetched in non-train mode or when
            # print_training_accuracy is set.
            if not phase_train or self.params.print_training_accuracy:
                logits = forward_pass_and_grad_outputs.pop(0)

            loss = (
                forward_pass_and_grad_outputs[0]
                if forward_pass_and_grad_outputs else None)
            grads = (
                forward_pass_and_grad_outputs[1:]
                if forward_pass_and_grad_outputs else None)

            return logits, loss, grads

        def make_results(logits, loss, grads):
            """Generate results based on logits, loss and grads."""
            results = {}  # The return value

            if logits is not None:
                results['logits'] = logits
                accuracy_ops = self.model.accuracy_function(input_list, logits)
                for name, op in accuracy_ops.items():
                    results['accuracy:' + name] = op

            if loss is not None:
                results['loss'] = loss

            if grads is not None:
                param_refs = self.variable_mgr.trainable_variables_on_device(
                    rel_device_num, abs_device_num, writable=True)
                results['gradvars'] = list(zip(grads, param_refs))

            return results

        with tf.device(self.devices[rel_device_num]):
            outputs = maybe_compile(forward_pass_and_gradients, self.params)
            logits, loss, grads = unpack_forward_pass_and_gradients_output(outputs)
            results_dict = make_results(logits, loss, grads)
            if self.my_params.need_record_internal_outputs:
                assert self.params.eval
                assert self.num_gpus == 1
                print('append the internal_outputs_dict to fetches')
                results_dict['internal_outputs'] = self.convnet_builder.get_internal_outputs_dict()
            #TODO shawn added this 20190313
            results_dict['labels'] = input_list[1]
            return results_dict


    #   overwrite this to customize the preprocessing
    def get_input_preprocessor(self):
        """Returns the image preprocessor to used, based on the model.

        Returns:
          The image preprocessor, or None if synthetic data should be used.
        """
        shift_ratio = 0
        if self.job_name:
            # shift_ratio prevents multiple workers from processing the same batch
            # during a step
            shift_ratio = float(self.task_index) / self.num_workers

        processor_class = self.dataset.get_input_preprocessor(
            self.params.input_preprocessor)
        assert processor_class
        return processor_class(
            self.batch_size * self.batch_group_size,
            self.model.get_input_shapes(),
            len(self.devices) * self.batch_group_size,
            dtype=self.model.data_type,
            train=(not self.params.eval),
            # TODO(laigd): refactor away image model specific parameters.
            distortions=self.params.distortions,
            resize_method=self.resize_method,
            shift_ratio=shift_ratio,
            summary_verbosity=self.params.summary_verbosity,
            distort_color_in_yiq=self.params.distort_color_in_yiq,
            fuse_decode_and_crop=self.params.fuse_decode_and_crop)

