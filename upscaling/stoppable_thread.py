import ctypes
import sys
import threading
import time


class StoppableThreadTimeout(TimeoutError):
    pass


class StoppableThreadExit(SystemExit):
    pass


class StoppableThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), daemon=True, timeout: int = 0, **kwargs):
        super().__init__(group, target, name, args, daemon=daemon, kwargs=kwargs)
        self.timeout = timeout
        self.begin_at = None

    def start(self):
        if self.timeout:
            self.begin_at = time.time()
            self.run = self.traced_run
        super().start()

    def traced_run(self):
        sys.settrace(self.globaltrace)
        super().run()

    def globaltrace(self, frame, event, arg):
        if event == 'call':
            return self.localtrace

    def localtrace(self, frame, event, arg):
        elapsed = time.time() - self.begin_at
        if elapsed > self.timeout and event == 'line':
            raise StoppableThreadTimeout(f'{self.name} timeout, {elapsed} > {self.timeout}')
        return self.localtrace

    def get_tid(self):
        for tid, thread in threading._active.items():
            if thread is self:
                return tid

    def terminate(self):
        if not (tid := self.get_tid()):
            print('No thread found')
            return
        num_modified = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(SystemExit))
        if num_modified > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise StoppableThreadExit('PyThreadState_SetAsyncExc failed')


# testing
if __name__ == '__main__':
    from datetime import datetime
    import traceback

    def experiment(id, **kwargs):
        print(datetime.now(), f'id={id}, kwargs={kwargs}. Experiment begins')
        for n in range(1, 101):
            print(datetime.now(), f'Iterate {n} time(s)')
            time.sleep(3)
        print('Experiment ends')

    try:
        print('Testing for KILL')
        for id in range(3):
            t = StoppableThread(target=experiment, args=(id,), a=1)
            t.start()
            time.sleep(1)
            t.terminate()
            t.join()

        print('Testing for TIMEOUT')
        for id in range(3):
            t = StoppableThread(target=experiment, args=(id,), a=1, timeout=1)
            t.start()
            t.join()
    # cannot catch exceptions of a thread in the main thread
    # except StoppableThreadTimeout as e:
    #     print(f'Timeout caught. {str(e)}')
    except:
        print(f'Main thread caught {traceback.format_exc()}')

    sys.exit()
