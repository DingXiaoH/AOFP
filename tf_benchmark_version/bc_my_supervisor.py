import tensorflow as tf
from tensorflow.train import *
from tensorflow.python.training.session_manager import *
from tensorflow.python.training.supervisor import *
from tensorflow.python.training.session_manager import _maybe_name

USE_DEFAULT = 0

class MySupervisor(Supervisor):

    def __init__(self,
                 graph=None,
                 ready_op=USE_DEFAULT,
                 ready_for_local_init_op=USE_DEFAULT,
                 is_chief=True,
                 init_op=USE_DEFAULT,
                 init_feed_dict=None,
                 local_init_op=USE_DEFAULT,
                 logdir=None,
                 summary_op=USE_DEFAULT,
                 saver=USE_DEFAULT,
                 global_step=USE_DEFAULT,
                 save_summaries_secs=120,
                 save_model_secs=600,
                 recovery_wait_secs=30,
                 stop_grace_secs=120,
                 checkpoint_basename="model.ckpt",
                 session_manager=None,
                 summary_writer=USE_DEFAULT,
                 init_fn=None,
                 local_init_run_options=None,
                 load_ckpt_full_path=None,
                 auto_continue=False,
                 should_write_graph=False):

        self.auto_continue = auto_continue

        super(MySupervisor, self).__init__(graph=graph,
            ready_op=ready_op,
            ready_for_local_init_op=ready_for_local_init_op,
            is_chief=is_chief,
            init_op=init_op,
            init_feed_dict=init_feed_dict,
            local_init_op=local_init_op,
            logdir=logdir,
            summary_op=summary_op,
            saver=saver,
            global_step=global_step,
            save_summaries_secs=save_summaries_secs,
            save_model_secs=save_model_secs,
            recovery_wait_secs=recovery_wait_secs,
            stop_grace_secs=stop_grace_secs,
            checkpoint_basename=checkpoint_basename,
            session_manager=session_manager,
            summary_writer=summary_writer,
            init_fn=init_fn,
            local_init_run_options=local_init_run_options,
        )
        self.load_ckpt_full_path = load_ckpt_full_path
        self.should_write_graph = should_write_graph

        print('MySupervisor initialized, the _logdir={}, _save_path={}, load_ckpt_full_path={}, auto_continue={}'.format(self._logdir, self._save_path, self.load_ckpt_full_path, self.auto_continue))


    def prepare_or_wait_for_session(self, master="", config=None,
                                    wait_for_checkpoint=False,
                                    max_wait_secs=7200,
                                    start_standard_services=True):
        """Make sure the model is ready to be used.

        Create a session on 'master', recovering or initializing the model as
        needed, or wait for a session to be ready.  If running as the chief
        and `start_standard_service` is set to True, also call the session
        manager to start the standard services.

        Args:
          master: name of the TensorFlow master to use.  See the `tf.Session`
            constructor for how this is interpreted.
          config: Optional ConfigProto proto used to configure the session,
            which is passed as-is to create the session.
          wait_for_checkpoint: Whether we should wait for the availability of a
            checkpoint before creating Session. Defaults to False.
          max_wait_secs: Maximum time to wait for the session to become available.
          start_standard_services: Whether to start the standard services and the
            queue runners.

        Returns:
          A Session object that can be used to drive the model.
        """
        # For users who recreate the session with prepare_or_wait_for_session(), we
        # need to clear the coordinator's stop_event so that threads managed by the
        # coordinator can run.
        self._coord.clear_stop()
        if self._summary_writer:
            self._summary_writer.reopen()

        if self._is_chief:
            sess = self._session_manager.prepare_session(
                master, init_op=self.init_op, saver=self.saver,
                checkpoint_dir=self._logdir, wait_for_checkpoint=wait_for_checkpoint,
                max_wait_secs=max_wait_secs, config=config,
                init_feed_dict=self._init_feed_dict, init_fn=self._init_fn, checkpoint_filename_with_path=self.load_ckpt_full_path)
            if self.should_write_graph:
                self._write_graph()
            if start_standard_services:
                logging.info("Starting standard services.")
                self.start_standard_services(sess)
        else:
            sess = self._session_manager.wait_for_session(master,
                config=config,
                max_wait_secs=max_wait_secs)
        if start_standard_services:
            logging.info("Starting queue runners.")
            self.start_queue_runners(sess)
        return sess

    def _init_session_manager(self, session_manager=None):
        if session_manager is None:
            self._session_manager = MySessionManager(
                local_init_op=self._local_init_op,
                ready_op=self._ready_op,
                ready_for_local_init_op=self._ready_for_local_init_op,
                graph=self._graph,
                recovery_wait_secs=self._recovery_wait_secs,
                local_init_run_options=self._local_init_run_options)
            if self.auto_continue:
                self._session_manager.allow_auto_continue()
        else:
            self._session_manager = session_manager


class MySessionManager(session_manager_mod.SessionManager):

    def allow_auto_continue(self):
        self.auto_continue = True

    def _restore_checkpoint(self,
                            master,
                            saver=None,
                            checkpoint_dir=None,
                            checkpoint_filename_with_path=None,
                            wait_for_checkpoint=False,
                            max_wait_secs=7200,
                            config=None):
        """Creates a `Session`, and tries to restore a checkpoint.


        Args:
          master: `String` representation of the TensorFlow master to use.
          saver: A `Saver` object used to restore a model.
          checkpoint_dir: Path to the checkpoint files. The latest checkpoint in the
            dir will be used to restore.
          checkpoint_filename_with_path: Full file name path to the checkpoint file.
          wait_for_checkpoint: Whether to wait for checkpoint to become available.
          max_wait_secs: Maximum time to wait for checkpoints to become available.
          config: Optional `ConfigProto` proto used to configure the session.

        Returns:
          A pair (sess, is_restored) where 'is_restored' is `True` if
          the session could be restored, `False` otherwise.

        Raises:
          ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
            set.
        """
        self._target = master
        sess = session.Session(self._target, graph=self._graph, config=config)

        if checkpoint_dir and checkpoint_filename_with_path:
            print('the supervisor got both checkpoint_dir={} and full_path={}, will restore from the latter'.format(checkpoint_dir, checkpoint_filename_with_path))

        # If either saver or checkpoint_* is not specified, cannot restore. Just
        # return.
        if not saver or not (checkpoint_dir or checkpoint_filename_with_path):
            return sess, False

        if checkpoint_filename_with_path:
            saver.restore(sess, checkpoint_filename_with_path)
            return sess, True

        if not hasattr(self, 'auto_continue'):
            print('auto_continue is not allowed, so no need to try to recover from the checkpoint')
            return sess, False

        # Waits up until max_wait_secs for checkpoint to become available.
        wait_time = 0
        ckpt = checkpoint_management.get_checkpoint_state(checkpoint_dir)
        while not ckpt or not ckpt.model_checkpoint_path:
            if wait_for_checkpoint and wait_time < max_wait_secs:
                logging.info("Waiting for checkpoint to be available.")
                time.sleep(self._recovery_wait_secs)
                wait_time += self._recovery_wait_secs
                ckpt = checkpoint_management.get_checkpoint_state(checkpoint_dir)
            else:
                return sess, False

        # Loads the checkpoint.
        saver.restore(sess, ckpt.model_checkpoint_path)
        saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
        return sess, True


    def prepare_session(self,
                        master,
                        init_op=None,
                        saver=None,
                        checkpoint_dir=None,
                        checkpoint_filename_with_path=None,
                        wait_for_checkpoint=False,
                        max_wait_secs=7200,
                        config=None,
                        init_feed_dict=None,
                        init_fn=None):
        """Creates a `Session`. Makes sure the model is ready to be used.

        Creates a `Session` on 'master'. If a `saver` object is passed in, and
        `checkpoint_dir` points to a directory containing valid checkpoint
        files, then it will try to recover the model from checkpoint. If
        no checkpoint files are available, and `wait_for_checkpoint` is
        `True`, then the process would check every `recovery_wait_secs`,
        up to `max_wait_secs`, for recovery to succeed.

        If the model cannot be recovered successfully then it is initialized by
        running the `init_op` and calling `init_fn` if they are provided.
        The `local_init_op` is also run after init_op and init_fn, regardless of
        whether the model was recovered successfully, but only if
        `ready_for_local_init_op` passes.

        If the model is recovered from a checkpoint it is assumed that all
        global variables have been initialized, in particular neither `init_op`
        nor `init_fn` will be executed.

        It is an error if the model cannot be recovered and no `init_op`
        or `init_fn` or `local_init_op` are passed.

        Args:
          master: `String` representation of the TensorFlow master to use.
          init_op: Optional `Operation` used to initialize the model.
          saver: A `Saver` object used to restore a model.
          checkpoint_dir: Path to the checkpoint files. The latest checkpoint in the
            dir will be used to restore.
          checkpoint_filename_with_path: Full file name path to the checkpoint file.
          wait_for_checkpoint: Whether to wait for checkpoint to become available.
          max_wait_secs: Maximum time to wait for checkpoints to become available.
          config: Optional `ConfigProto` proto used to configure the session.
          init_feed_dict: Optional dictionary that maps `Tensor` objects to feed
            values.  This feed dictionary is passed to the session `run()` call when
            running the init op.
          init_fn: Optional callable used to initialize the model. Called after the
            optional `init_op` is called.  The callable must accept one argument,
            the session being initialized.

        Returns:
          A `Session` object that can be used to drive the model.

        Raises:
          RuntimeError: If the model cannot be initialized or recovered.
          ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
            set.
        """

        sess, is_loaded_from_checkpoint = self._restore_checkpoint(
            master,
            saver,
            checkpoint_dir=checkpoint_dir,
            checkpoint_filename_with_path=checkpoint_filename_with_path,
            wait_for_checkpoint=wait_for_checkpoint,
            max_wait_secs=max_wait_secs,
            config=config)
        print('is_loaded_from_checkpoint:', is_loaded_from_checkpoint)
        if not is_loaded_from_checkpoint:
            if init_op is None and not init_fn and self._local_init_op is None:
                raise RuntimeError("Model is not initialized and no init_op or "
                                   "init_fn or local_init_op was given")
            if init_op is not None:
                print('running the init_op')
                sess.run(init_op, feed_dict=init_feed_dict)
                print('done the init_op')
            if init_fn:
                init_fn(sess)

        local_init_success, msg = self._try_run_local_init_op(sess)
        if not local_init_success:
            raise RuntimeError(
                "Init operations did not make model ready for local_init.  "
                "Init op: %s, init fn: %s, error: %s" % (_maybe_name(init_op),
                                                         init_fn,
                                                         msg))
        print('done local_init_op')

        is_ready, msg = self._model_ready(sess)
        if not is_ready:
            raise RuntimeError(
                "Init operations did not make model ready.  "
                "Init op: %s, init fn: %s, local_init_op: %s, error: %s" %
                (_maybe_name(init_op), init_fn, self._local_init_op, msg))
        print('model ready')
        return sess


