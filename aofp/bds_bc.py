
from bc import *
from aofp.bds_utils import calc_next_half_size, generate_base_mask_feed_dict, generate_ds_mask_feed_dict
from bc_helpers import _NUM_STEPS_TO_PROFILE
from bc_base import *
from bc_constants import MASK_VALUE_KEYWORD
from aofp.bds_convnetbuilder import BDSConvNetBuilder




def bds_train_one_step(sess,
                       fetches,
                       step,
                       batch_size,
                       step_train_times,
                       trace_filename,
                       partitioned_graph_file_prefix,
                       profiler,
                       image_producer,
                       params,
                       summary_op=None,
                       show_images_per_sec=True,
                       benchmark_logger=None,
                       collective_graph_key=0,
                       track_mvav_op=None,
                       metric_dict=None,
                       feed_dict=None):
    """Advance one step of benchmarking."""
    should_profile = profiler and 0 <= step < _NUM_STEPS_TO_PROFILE
    need_options_and_metadata = (
        should_profile or collective_graph_key > 0 or
        ((trace_filename or partitioned_graph_file_prefix) and step == -2)
    )
    if need_options_and_metadata:
        run_options = tf.RunOptions()
        if (trace_filename and step == -2) or should_profile:
            run_options.trace_level = tf.RunOptions.FULL_TRACE
        if partitioned_graph_file_prefix and step == -2:
            run_options.output_partition_graphs = True
        if collective_graph_key > 0:
            run_options.experimental.collective_graph_key = collective_graph_key
        run_metadata = tf.RunMetadata()
    else:
        run_options = None
        run_metadata = None
    summary_str = None
    start_time = time.time()


    if track_mvav_op is None:
        if summary_op is None:
            results, metrics = sess.run([fetches, metric_dict], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
        else:
            (results, summary_str, metrics) = sess.run(
                [fetches, summary_op, metric_dict], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
    else:
        if summary_op is None:
            results, _, metrics = sess.run([fetches, track_mvav_op, metric_dict], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
        else:
            (results, summary_str, _, metrics) = sess.run(
                [fetches, summary_op, track_mvav_op, metric_dict], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)

    if not params.forward_only:
        lossval = results['average_loss']
    else:
        lossval = 0.
    if image_producer is not None:
        image_producer.notify_image_consumption()
    train_time = time.time() - start_time
    step_train_times.append(train_time)
    if (show_images_per_sec and step >= 0 and
            (step == 0 or (step + 1) % params.display_every == 0)):
        speed_mean, speed_uncertainty, speed_jitter = get_perf_timing(
            batch_size, step_train_times)
        log_str = '%i\t%s\t%.*f' % (
            step + 1,
            get_perf_timing_str(speed_mean, speed_uncertainty, speed_jitter),
            LOSS_AND_ACCURACY_DIGITS_TO_SHOW, lossval)
        if 'top_1_accuracy' in results:
            log_str += '\t%.*f\t%.*f' % (
                LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_1_accuracy'],
                LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_5_accuracy'])
        log_fn(log_str)
        if benchmark_logger:
            benchmark_logger.log_metric(
                'current_examples_per_sec', speed_mean, global_step=step + 1)
            if 'top_1_accuracy' in results:
                benchmark_logger.log_metric(
                    'top_1_accuracy', results['top_1_accuracy'], global_step=step + 1)
                benchmark_logger.log_metric(
                    'top_5_accuracy', results['top_5_accuracy'], global_step=step + 1)
    if need_options_and_metadata:
        if should_profile:
            profiler.add_step(step, run_metadata)
        if trace_filename and step == -2:
            log_fn('Dumping trace to %s' % trace_filename)
            trace_dir = os.path.dirname(trace_filename)
            if not gfile.Exists(trace_dir):
                gfile.MakeDirs(trace_dir)
            with gfile.Open(trace_filename, 'w') as trace_file:
                if params.use_chrome_trace_format:
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
                else:
                    trace_file.write(str(run_metadata.step_stats))
        if partitioned_graph_file_prefix and step == -2:
            path, filename = os.path.split(partitioned_graph_file_prefix)
            if '.' in filename:
                base_filename, ext = filename.rsplit('.', 1)
                ext = '.' + ext
            else:
                base_filename, ext = filename, ''
            as_text = filename.endswith('txt')
            for graph_def in run_metadata.partition_graphs:
                device = graph_def.node[0].device.replace('/', '_').replace(':', '_')
                graph_filename = '%s%s%s' % (base_filename, device, ext)
                log_fn('Writing partitioned GraphDef as %s to %s' % (
                    'text' if as_text else 'binary',
                    os.path.join(path, graph_filename)))
                tf.train.write_graph(graph_def, path, graph_filename, as_text)
    return (summary_str, lossval, metrics)


class BDSBenchmark(BenchmarkCNN):

    def __init__(self, params, my_params, bds_params, dataset=None, model=None):
        super(BDSBenchmark, self).__init__(params=params, my_params=my_params, dataset=dataset, model=model)
        self.bds_params = bds_params
        self.ds_masks = []
        self.base_masks = []
        self.bds_metrics = []

    def _get_mask_dict(self, layer_to_base_mask_value):
        mask_dict = {}
        for i, v in layer_to_base_mask_value.items():
            mask_dict[MASK_VALUE_KEYWORD + str(i)] = v
        return mask_dict

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
        convnet_builder = BDSConvNetBuilder(images, input_nchan=3, phase_train=phase_train, use_tf_layers=self.params.use_tf_layers,
            data_format=self.params.data_format, bds_params=self.bds_params, dtype=data_type, variable_dtype=var_type, use_dense_layer=self.my_params.use_dense_layer)
        return convnet_builder


    def postprocess_after_build_by_convnet_builder(self, convnet_builder, build_results):
        self.ds_masks.append(convnet_builder.get_ds_masks())
        self.base_masks.append(convnet_builder.get_base_masks())
        self.bds_metrics.append(convnet_builder.get_bds_metrics())


    def save_hdf5_and_masks_by_global_step(self, cur_global_step, layer_to_base_mask_value):
        save_path = os.path.join(self.params.train_dir, 'ckpt_step_{}.hdf5'.format(cur_global_step))
        print('saving hdf5 to ', save_path)
        self.save_weights_and_extra(save_path, self._get_mask_dict(layer_to_base_mask_value))
        if self.my_params.save_mvav:
            self.save_moving_average_weights_to_hdf5(save_path.replace('.hdf5', '_mvav.hdf5'),
                moving_averages=self.variable_averages)

    def bds_seek_by_loss(self, granu, batches_per_half, start_mask, log_file, exp_layer_idx, use_half_dropout=True, drop_score_path=False):

        sess = self.sess
        print('showing the current base masks:', self.base_masks)
        print('showing the current scoring masks:', self.ds_masks)

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

        cur_feed_dict = generate_base_mask_feed_dict(self, exp_layer_idx, start_mask)
        if drop_score_path:
            target_mask = self.ds_masks[0][exp_layer_idx]
        else:
            target_mask = self.base_masks[0][exp_layer_idx]
        excluded = start_mask == 0
        half_idx = 0

        while True:
            print('current excluded: ', excluded)
            cur_search_space = np.where(excluded == False)[0]  # original filter index space
            cur_search_space_size = len(cur_search_space)
            target_half_size = calc_next_half_size(cur_search_space_size, granu)

            log_important('one half started. halve from {} to {}'.format(cur_search_space_size, target_half_size), log_file)

            absent_losses = [[] for i in range(cur_search_space_size)]  # search index space

            total_loss_sum = 0.0

            start_time = time.time()
            if granu > 1:
                nb_drop = granu
            elif use_half_dropout:
                nb_drop = cur_search_space_size // 2
            else:
                nb_drop = 1
            print('start a half with nb_drop=', nb_drop)
            for step in xrange(batches_per_half):
                #####################
                drop_choice = np.random.choice(cur_search_space_size, nb_drop,
                    replace=False)  # search index space

                drop_filter_idxes = cur_search_space[drop_choice]  # original filter index space
                for k in drop_filter_idxes:
                    assert not excluded[k]
                mask_value = np.array(start_mask)
                mask_value[drop_filter_idxes] = 0
                cur_feed_dict[target_mask] = mask_value
                #####################
                results = sess.run(self.fetches, feed_dict=cur_feed_dict)
                results = self.model.postprocess(results)
                total_loss_sum += results['loss']
                if (step + 1) % self.params.display_every == 0:
                    duration = time.time() - start_time
                    examples_per_sec = (
                        self.batch_size * self.params.display_every / duration)
                    log_fn('%i\t%.1f examples/sec' % (step + 1, examples_per_sec))
                    start_time = time.time()
                if image_producer is not None:
                    image_producer.notify_image_consumption()
                # TODO
                metric = results['loss']
                for i in drop_choice:
                    absent_losses[i].append(metric)

            # one move is finished
            # Compute statistiacs
            mean_loss = total_loss_sum / batches_per_half
            # compute loss increase ratio
            avg_absent_loss_vector = np.zeros((cur_search_space_size,))
            for i in range(cur_search_space_size):
                absent_loss_avg = np.mean(np.array(absent_losses[i]))
                avg_absent_loss_vector[i] = absent_loss_avg

            # cur_min_loss_increase = np.min(avg_absent_loss_vector)
            passed_idxes = np.argsort(avg_absent_loss_vector)[:target_half_size]
            blocked_idxes = np.delete(np.arange(cur_search_space_size), passed_idxes)

            log_important(
                'half finished, average loss {}'.format(mean_loss), log_file)
            log_important('newly excluded {} filters'.format(len(blocked_idxes)), log_file)

            if target_half_size == granu:
                return cur_search_space[passed_idxes]
            elif target_half_size < granu:
                assert False
            excluded[cur_search_space[blocked_idxes]] = True
            half_idx += 1



    def bds_seek_by_shortcut(self, granu, batches_per_half, start_mask, log_file, exp_layer_idx):

        sess = self.sess

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

        target_mask = self.ds_masks[exp_layer_idx]
        cur_feed_dict = generate_ds_mask_feed_dict(self, exp_layer_idx, start_mask)
        excluded = start_mask == 0
        half_idx = 0

        ############
        to_fetch = self.bds_metrics[exp_layer_idx]
        ############

        while True:
            cur_search_space = np.where(excluded == False)[0]  # original filter index space
            cur_search_space_size = len(cur_search_space)
            target_half_size = calc_next_half_size(cur_search_space_size, granu)

            log_important('one half started. halve from {} to {}'.format(cur_search_space_size, target_half_size), log_file)

            absent_losses = [[] for i in range(cur_search_space_size)]  # search index space

            total_top1_sum = 0.0
            total_top5_sum = 0.0
            total_loss_sum = 0.0

            start_time = time.time()
            for step in xrange(batches_per_half):
                #####################
                drop_choice = np.random.choice(cur_search_space_size, cur_search_space_size // 2,
                    replace=False)  # search index space
                drop_filter_idxes = cur_search_space[drop_choice]  # original filter index space
                mask_value = np.array(start_mask)
                mask_value[drop_filter_idxes] = 0
                cur_feed_dict[target_mask] = mask_value
                #####################
                results = sess.run(to_fetch, feed_dict=cur_feed_dict)

                if (step + 1) % self.params.display_every == 0:
                    duration = time.time() - start_time
                    examples_per_sec = (
                        self.batch_size * self.params.display_every / duration)
                    log_fn('%i\t%.1f examples/sec' % (step + 1, examples_per_sec))
                    start_time = time.time()
                if image_producer is not None:
                    image_producer.notify_image_consumption()
                # TODO
                metric = results
                for i in drop_choice:
                    absent_losses[i].append(metric)


            # one move is finished
            # Compute statistiacs
            accuracy_at_1 = total_top1_sum / batches_per_half
            accuracy_at_5 = total_top5_sum / batches_per_half
            mean_loss = total_loss_sum / batches_per_half
            # compute loss increase ratio
            avg_absent_loss_vector = np.zeros((cur_search_space_size,))
            for i in range(cur_search_space_size):
                absent_loss_avg = np.mean(np.array(absent_losses[i]))
                avg_absent_loss_vector[i] = absent_loss_avg

            # cur_min_loss_increase = np.min(avg_absent_loss_vector)
            passed_idxes = np.argsort(avg_absent_loss_vector)[:target_half_size]
            blocked_idxes = np.delete(np.arange(cur_search_space_size), passed_idxes)

            log_important(
                'half finished, top1 {}, top5 {}, average loss {}'.format(accuracy_at_1,
                    accuracy_at_5,
                    mean_loss), log_file)
            log_important('newly excluded {} filters'.format(len(blocked_idxes)), log_file)

            if target_half_size == granu:
                return cur_search_space[passed_idxes]
            elif target_half_size < granu:
                assert False
            excluded[cur_search_space[blocked_idxes]] = True
            half_idx += 1


    def _init_from_hdf5_vars_and_values(self, hdf5_file):
        value_ignore_patterns = ['tower_[0-9]/', ':0', 'cg/', 'conv2d/', re.compile('batchnorm(\d+)/'), '/ExponentialMovingAverage']
        var_ignore_patterns = ['tower_[0-9]/', ':0', 'cg/', 'conv2d/', re.compile('batchnorm(\d+)/')]
        self.init_hdf5 = hdf5_file
        vars = self.get_key_variables()
        _dic = read_hdf5(hdf5_file)
        dic = {}
        for k, v in _dic.items():
            dic[eliminate_all_patterns_and_starting_vs(k, value_ignore_patterns)] = v
        tensors = []
        values = []
        for t in vars:
            name = eliminate_all_patterns_and_starting_vs(t.name, var_ignore_patterns)
            if name in dic:
                tensors.append(t)
                values.append(dic[name])
                print('ready to load: ', name, t.get_shape(), dic[name].shape)
                # print(name)
            else:
                print('cannot find matched value for variable ', name)
        print('loaded hdf5, {} matched key vars and {} values'.format(len(tensors), len(values)))

        self._init_expma_vars(dic, tensors, values, var_ignore_patterns)

        found_masks = {}
        for n, v in dic.items():
            if MASK_VALUE_KEYWORD in n:
                found_masks[int(n.replace(MASK_VALUE_KEYWORD, ''))] = v
        print('-------------found {} mask values in {}, they are -------------'.format(len(found_masks), self.my_params.init_hdf5))
        for i, v in found_masks.items():
            print('layer{}, remain{}'.format(i, np.sum(v)))
        print('-------------done display-----------')
        #   not found, initialize
        for i in range(len(self.params.deps)):
            if i not in found_masks:
                found_masks[i] = np.ones(int(self.params.deps[i]), dtype=np.float32)
        self.layer_to_mask_value = found_masks

        return tensors, values



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


            local_step = 0
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



            print('self.lr_boundaries=', self.lr_boundaries)


            ########################    TODO the following are bds-related

            tower_to_layer_to_metric_ops = self.bds_metrics
            bds_layers = tower_to_layer_to_metric_ops[0].keys()

            tower_to_layer_to_base_mask_ph = self.base_masks
            tower_to_layer_to_ds_mask_ph = self.ds_masks
            num_replicas = len(tower_to_layer_to_base_mask_ph)

            INIT_GRANU = 99999

            #   bds state information
            layer_to_move_idx = {i:0 for i in bds_layers}
            layer_to_half_idx = {i:0 for i in bds_layers}

            #   initialize the start_local_step evenly
            _init_interval = int(self.bds_params.train_batches_per_half / len(bds_layers))
            layer_to_half_start_local_step = {i:(_init_interval * i + self.bds_params.bds_start_step) for i in bds_layers}
            print('the initial layer_to_half_start_local_step={}'.format(layer_to_half_start_local_step))
            layer_to_base_mask_value = self.layer_to_mask_value   # changed after each move, # TODO pay attention to save and load
            layer_to_search_space = {i:(np.where(layer_to_base_mask_value[i] > 0.0)[0]) for i in bds_layers}

            overall_num_moves = 0

            layer_to_absent_metrics = {}
            for i in bds_layers:
                filter_absent_metrics = list()
                layer_to_absent_metrics[i] = filter_absent_metrics


            def _reset_layer_to_absent_metrics(layer_idx):
                layer_to_absent_metrics[layer_idx].clear()
                for i in range(0, len(layer_to_search_space[layer_idx])):
                    layer_to_absent_metrics[layer_idx].append(list())

            #   should be called at the very beginning
            def prepare_for_a_move(layer_idx, local_step):
                layer_to_half_idx[layer_idx] = 0
                layer_to_half_start_local_step[layer_idx] = max(local_step, layer_to_half_start_local_step[layer_idx])
                #   reset layer_to_search_space by layer_to_base_mask_value, which should be modified by last move
                layer_to_search_space[layer_idx] = np.where(layer_to_base_mask_value[layer_idx] > 0.0)[0]
                _reset_layer_to_absent_metrics(layer_idx)

            def move_to_next_move(layer_idx, local_step, picked):   # picked: recently pruned
                #   TODO save?
                for p in picked:
                    layer_to_base_mask_value[layer_idx][p] = 0.0
                layer_to_move_idx[layer_idx] += 1
                prepare_for_a_move(layer_idx, local_step)
                if overall_num_moves % self.bds_params.save_per_moves == 0:
                    cur_global_step = sess.run(graph_info.global_step)
                    self.save_hdf5_and_masks_by_global_step(cur_global_step=cur_global_step, layer_to_base_mask_value=layer_to_base_mask_value)



            def quit_move(layer_idx, local_step):
                #   TODO should do what else?
                prepare_for_a_move(layer_idx, local_step)

            def refine_to_next_half(layer_idx, local_step, excluded_filter_ids):
                layer_to_half_idx[layer_idx] += 1
                layer_to_half_start_local_step[layer_idx] = local_step
                #   remove recently excluded filters from the search space
                new_search_space = [k for k in layer_to_search_space[layer_idx] if k not in excluded_filter_ids]
                layer_to_search_space[layer_idx] = np.array(new_search_space)
                _reset_layer_to_absent_metrics(layer_idx)

            #   generate [feed_dict, layer_to_drop_filter_ids]  by  [layer_to_base_mask_value, layer_to_search_space]
            def prepare_for_a_step():
                feed_dict = {}
                tower_to_layer_to_drop_filter_ids = {k:dict() for k in range(num_replicas)}
                tower_to_layer_to_drop_in_search_space = {k:dict() for k in range(num_replicas)}
                for i in bds_layers:
                    cur_search_space_size = len(layer_to_search_space[i])
                    if self.bds_params.use_single_dropout:
                        nb_drop = 1
                    else:
                        nb_drop = cur_search_space_size // 2
                    for k in range(num_replicas):
                        drop_choice = np.random.choice(cur_search_space_size, nb_drop,
                            replace=False).astype(np.int32)  # search index space
                        tower_to_layer_to_drop_in_search_space[k][i] = drop_choice
                        drop_filter_idxes = layer_to_search_space[i][drop_choice]  # original filter index space
                        tower_to_layer_to_drop_filter_ids[k][i] = drop_filter_idxes
                        ds_mask_value = np.array(layer_to_base_mask_value[i])
                        ds_mask_value[drop_filter_idxes] = 0
                        feed_dict[tower_to_layer_to_base_mask_ph[k][i]] = layer_to_base_mask_value[i]
                        feed_dict[tower_to_layer_to_ds_mask_ph[k][i]] = ds_mask_value
                return feed_dict, tower_to_layer_to_drop_filter_ids, tower_to_layer_to_drop_in_search_space


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
                    print(len(step_train_times))
                    print(self.num_warmup_batches)
                    # assert len(step_train_times) == self.num_warmup_batches
                    # reset times to ignore warm up batch
                    step_train_times = []
                    loop_start_time = time.time()

                    ############    TODO initialize, need to save? I think this is no need to save local_step
                    for i in bds_layers:
                        prepare_for_a_move(i, local_step=0)


                if (summary_writer and (local_step + 1) % self.params.save_summaries_steps == 0):
                    fetch_summary = summary_op
                else:
                    fetch_summary = None
                collective_graph_key = 7 if (self.params.variable_update == 'collective_all_reduce') else 0


                ########### TODO note
                cur_feed_dict, cur_tower_to_layer_to_drop_ids, cur_tower_to_layer_to_drop_in_search_space = prepare_for_a_step()

                (summary_str, last_average_loss, metric_values) = bds_train_one_step(
                    sess, graph_info.fetches, local_step,
                    self.batch_size * (self.num_workers
                                       if self.single_session else 1), step_train_times,
                    self.trace_filename, self.params.partitioned_graph_file_prefix,
                    profiler, image_producer, self.params, fetch_summary,
                    benchmark_logger=self.benchmark_logger,
                    collective_graph_key=collective_graph_key,
                    track_mvav_op=graph_info.mvav_op,
                    metric_dict=tower_to_layer_to_metric_ops,
                    feed_dict=cur_feed_dict)

                local_step += 1

                termination_satisfied = False
                for i in bds_layers:
                    for p in range(num_replicas):
                        for drop_filter in cur_tower_to_layer_to_drop_in_search_space[p][i]:
                            layer_to_absent_metrics[i][drop_filter].append(metric_values[p][i])
                    #   check if a half finished
                    if local_step - layer_to_half_start_local_step[i] < self.bds_params.train_batches_per_half:
                        continue
                    # one half is finished,  calculate the mean metric
                    cur_search_space_size = len(layer_to_search_space[i])
                    assert cur_search_space_size <= self.params.deps[i]
                    #   only one filter remains? skip
                    if cur_search_space_size == 1:
                        continue

                    absent_avg_metric_vector = np.zeros(cur_search_space_size)
                    for k in range(cur_search_space_size):
                        if len(layer_to_absent_metrics[i][k]) > 0:
                            absent_avg = np.mean(np.array(layer_to_absent_metrics[i][k]))
                        else:
                            absent_avg = 0
                        absent_avg_metric_vector[k] = absent_avg

                    #   get the next half size
                    if layer_to_half_idx[i] == 0:
                        target_search_space_size = calc_next_half_size(cur_search_space_size, INIT_GRANU)
                    else:
                        target_search_space_size = cur_search_space_size // 2

                    passed_in_search_space = np.argsort(absent_avg_metric_vector)[:target_search_space_size]
                    blocked_in_search_space = np.delete(np.arange(cur_search_space_size), passed_in_search_space)

                    max_selected_metric = absent_avg_metric_vector[passed_in_search_space[-1]]

                    passed_filter_ids = layer_to_search_space[i][passed_in_search_space]
                    blocked_filter_ids = layer_to_search_space[i][blocked_in_search_space]

                    log_important(
                        'layer {}: one half finished, {} passed, {} blocked'.format(i, len(passed_filter_ids), len(blocked_filter_ids)), self.bds_params.bds_log_file)

                    if max_selected_metric <= self.bds_params.inc_ratio_limit:

                        log_important('layer {}: max metric {} < limit {}, pruned {}, move to next move'
                            .format(i, max_selected_metric, self.bds_params.inc_ratio_limit, list(passed_filter_ids)), self.bds_params.bds_log_file)
                        overall_num_moves += 1
                        #   TODO add summary for the change of num_fitlers
                        summary = tf.Summary()
                        summary.value.add(tag='remain_layer{}'.format(i), simple_value=np.sum(layer_to_base_mask_value[i]) - len(passed_filter_ids))
                        cur_full_deps = np.array(self.params.deps)
                        for iii in bds_layers:
                            cur_full_deps[iii] = int(np.sum(layer_to_base_mask_value[iii]))
                        origin_flops = np.sum(self.bds_params.flops_pruned_calc_fn(self.bds_params.flops_calc_baseline_deps))
                        cur_flops = np.sum(self.bds_params.flops_pruned_calc_fn(cur_full_deps))
                        pruned_flops = (1.0 - cur_flops / origin_flops)
                        summary.value.add(tag='flops_pruned', simple_value=pruned_flops)
                        summary_writer.add_summary(summary, sess.run(graph_info.global_step))
                        if pruned_flops > self.bds_params.terminate_at_pruned_flops:
                            termination_satisfied = True


                        move_to_next_move(layer_idx=i, local_step=local_step, picked=passed_filter_ids)

                    elif target_search_space_size <= 1:

                        log_important('layer {}: max metric {} > limit {} even when reached min granu {}, refining ended'
                                .format(i, max_selected_metric, self.bds_params.inc_ratio_limit, target_search_space_size), self.bds_params.bds_log_file)
                        quit_move(layer_idx=i, local_step=local_step)

                    else:
                        log_important('layer {}: max metric {} > limit {}, continue refining'
                                .format(i, max_selected_metric, self.bds_params.inc_ratio_limit), self.bds_params.bds_log_file)
                        refine_to_next_half(layer_idx=i, local_step=local_step, excluded_filter_ids=blocked_filter_ids)

            #############################################

                if summary_str is not None and is_chief:
                    sv.summary_computed(sess, summary_str)


                if local_step % 5000 == 0:
                    cur_remain_dep_list = []
                    for i in sorted(list(bds_layers)):
                        cur_remain_dep_list.append(np.sum(layer_to_base_mask_value[i]))
                    log_important('**********the remain deps at global step {} is {}'.format(sess.run(graph_info.global_step), cur_remain_dep_list), self.bds_params.bds_log_file)

                if (self.my_params.num_steps_per_hdf5 > 0 and local_step % self.my_params.num_steps_per_hdf5 == 0 and local_step > 0 and is_chief):
                    self.save_hdf5_and_masks_by_global_step(sess.run(graph_info.global_step), layer_to_base_mask_value=layer_to_base_mask_value)

                if (self.params.save_model_steps and local_step % self.params.save_model_steps == 0 and local_step > 0 and is_chief):
                    sv.saver.save(sess, sv.save_path, sv.global_step)

                if self.lr_boundaries is not None and local_step % 100 == 0 and local_step > 0 and is_chief:
                    cur_global_step = sess.run(graph_info.global_step)
                    for b in self.lr_boundaries:
                        if b > cur_global_step and b - cur_global_step < 100:
                            sv.saver.save(sess, sv.save_path, sv.global_step)
                            self.save_hdf5_and_masks_by_global_step(cur_global_step, layer_to_base_mask_value=layer_to_base_mask_value)
                            break

                if self.my_params.frequently_save_interval is not None and self.my_params.frequently_save_last_epochs is not None and local_step % self.my_params.frequently_save_interval == 0 and local_step > 0 and is_chief:
                    cur_global_step = sess.run(graph_info.global_step)
                    remain_steps = self.num_batches - cur_global_step
                    remain_epochs = remain_steps * self.batch_size / self.dataset.num_examples_per_epoch(self.subset)
                    if remain_epochs < self.my_params.frequently_save_last_epochs:
                        self.save_hdf5_and_masks_by_global_step(cur_global_step, layer_to_base_mask_value=layer_to_base_mask_value)

                if termination_satisfied:
                    break

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
                self.save_weights_and_extra(self.my_params.save_hdf5, self._get_mask_dict(layer_to_base_mask_value))

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

    def _generate_mask_feed_dict(self):
        feed_dict = {}
        for l, ph in self.base_masks[0].items():
            feed_dict[ph] = self.layer_to_mask_value[l]
            if hasattr(self, 'tmp_ds_mask_value'):      #TODO really ugly
                feed_dict[self.ds_masks[0][l]] = self.tmp_ds_mask_value  # useless. can be removed?
            else:
                feed_dict[self.ds_masks[0][l]] = self.layer_to_mask_value[l]  # useless. can be removed?
        return feed_dict

    def eval_once(self, saver, summary_writer, target, local_var_init_op_group,
                  input_producer_op, enqueue_ops, fetches, summary_op, eval_feed_dict=None):
        assert eval_feed_dict is None
        super(BDSBenchmark, self).eval_once(saver, summary_writer, target, local_var_init_op_group,
                  input_producer_op, enqueue_ops, fetches, summary_op, eval_feed_dict=self._generate_mask_feed_dict())

    def simple_eval(self, eval_record_comment, other_log_file=None, eval_feed_dict=None):
        if eval_feed_dict is None:
            eval_feed_dict = self._generate_mask_feed_dict()
        return super(BDSBenchmark, self).simple_eval(eval_record_comment=eval_record_comment, other_log_file=other_log_file,
            eval_feed_dict=self._generate_mask_feed_dict())

    def simple_survey(self, survey_save_file, survey_layer_ids, bias_opt, agg_opt, eval_feed_dict=None):
        if eval_feed_dict is None:
            eval_feed_dict = self._generate_mask_feed_dict()
        return super(BDSBenchmark, self).simple_survey(survey_save_file, survey_layer_ids, bias_opt,
            agg_opt, eval_feed_dict=eval_feed_dict)

    def get_deps_by_mask_value(self):
        layer_idx_to_dep = {}
        for l, v in self.layer_to_mask_value.items():
            layer_idx_to_dep[l] = int(np.sum(v > 0.99))
        deps_list = []
        for k in sorted(layer_idx_to_dep.keys()):
            deps_list.append(layer_idx_to_dep[k])
        return np.array(deps_list, dtype=np.int32)






