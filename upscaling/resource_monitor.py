import csv
from datetime import datetime
from pathlib import Path
import psutil
import time

from gpustat import GPUStatCollection
from upscaling.stoppable_thread import StoppableThread


class ResourceMonitor:
    def __init__(self):
        self.thread = None

    def start(self, file_path, monitoring_cycle=1):
        if not self.thread:
            self.thread = StoppableThread(target=self.monitor,
                                          args=(str(file_path) if isinstance(file_path, Path) else file_path,
                                                monitoring_cycle))
            self.thread.start()

    def monitor(self, file_path, monitoring_cycle):
        print(datetime.now(), f'Monitoring starts. Saving to {file_path}')

        with open(file_path, 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            header = ['Checked at', 'CPU', 'VMem', 'SMem', 'Read', 'Write', 'GPU', 'GMem', ]
            writer.writerow(header)

            while True:
                vmem = psutil.virtual_memory()
                smem = psutil.swap_memory()
                disk_io = psutil.disk_io_counters()
                gstat = GPUStatCollection.new_query().jsonify()['gpus'][0]
                usage = [datetime.now(),
                         psutil.cpu_percent(), vmem.used / 1024 / 1024, smem.used / 1024 / 1024,
                         disk_io.read_bytes / 1024 / 1024, disk_io.write_bytes / 1024 / 1024,
                         gstat['utilization.gpu'], gstat['memory.used']]
                writer.writerow(usage)
                output_file.flush()
                time.sleep(monitoring_cycle)

    def stop(self):
        if self.thread:
            self.thread.terminate()
            self.thread.join()
        print(datetime.now(), f'Monitoring stopped.')


# testing
if __name__ == '__main__':
    output_file = Path(__file__).parent.parent.joinpath('data', 'monitor_testing.csv')
    monitor = ResourceMonitor(output_file)
    monitor.start()
    time.sleep(3)
    monitor.stop()
