import datetime
import time
import shutil
from unittest import TestCase

import cv2
import numpy as np

from upscaling import project_home
from upscaling.nafnet.nafnet import Debluring, SuperResolution
from upscaling.resource_monitor import ResourceMonitor


class NAFNetTest(TestCase):
    @classmethod
    def setUpClass(cls):
        # test setting
        cls.scaling_rate = 4
        cls.reporting_cycle = 10

        # input files
        data_home = project_home.joinpath('data')
        input_dir = data_home.joinpath('i')
        print('Input folder=', input_dir)
        cls.input_files = [f for f in input_dir.iterdir() if f.is_file()]
        cls.num_input_files = len(cls.input_files)

        # output folder
        cls.output_dir = data_home.joinpath('nafnet')
        print('Output folder=', cls.output_dir)
        shutil.rmtree(cls.output_dir, ignore_errors=True)
        cls.output_dir.mkdir()

        cls.monitor = ResourceMonitor()

    def setUp(self):
        self.elapsed = 0
        self.last_reported = time.time()

    def tearDown(self):
        NAFNetTest.monitor.stop()

    # by debluring model
    def test001_deblur(self):
        print(datetime.datetime.now(), 'De-bluring starts')
        NAFNetTest.monitor.start(file_path=str(NAFNetTest.output_dir.joinpath('de-blur.csv')))
        deblurer = Debluring()

        for input_file_idx, input_file in enumerate(NAFNetTest.input_files):
            img = cv2.imread(str(input_file))
            assert img is not None, f'Cannot open {input_file}'
            h_resized, w_resized = (s * NAFNetTest.scaling_rate for s in img.shape[:2])

            cv2_resized = cv2.resize(img, (w_resized, h_resized))
            assert cv2_resized is not None, f'OpenCV resizing fails for {input_file}'
            start = time.time()
            deblured = deblurer.deblur(cv2_resized)
            assert deblured is not None, f'Debluring fails for {input_file}'
            self.elapsed += time.time() - start

            if time.time() - self.last_reported >= NAFNetTest.reporting_cycle:
                print(datetime.datetime.now(), f'{input_file_idx}/{NAFNetTest.num_input_files} '
                                               f'({input_file_idx / NAFNetTest.num_input_files * 100:.2f}%)')
                self.last_reported = time.time()
            break

        print('Elapsed=', self.elapsed, ', Elapsed per file=', self.elapsed / NAFNetTest.num_input_files)
        print(datetime.datetime.now(), 'De-bluring finished')

    # by super-resolution model
    def test002_upscale(self):
        print(datetime.datetime.now(), 'Upscaling starts')
        NAFNetTest.monitor.start(file_path=str(NAFNetTest.output_dir.joinpath('upscale.csv')))
        upscaler = SuperResolution()

        for input_file_idx, input_file in enumerate(NAFNetTest.input_files):
            img = cv2.imread(str(input_file))
            assert img is not None, f'Cannot open {input_file}'

            start = time.time()
            upscaled = upscaler.upscale(img)
            assert upscaled is not None, f'Upscaling fails for {input_file}'
            self.elapsed += time.time() - start

            if time.time() - self.last_reported >= NAFNetTest.reporting_cycle:
                print(datetime.datetime.now(), f'{input_file_idx}/{NAFNetTest.num_input_files} '
                                               f'({input_file_idx / NAFNetTest.num_input_files * 100:.2f}%)')
                self.last_reported = time.time()
            break

        print('Elapsed=', self.elapsed, ', Elapsed per file=', self.elapsed / NAFNetTest.num_input_files)
        print(datetime.datetime.now(), 'Upscaling finished')

    @classmethod
    def cls_dummy(cls):
        print('cls_dummy called')

    def instance_dummy(self):
        print('instance_dummy called')

    # def test003_compare(self):
    #
    #     print(datetime.datetime.now(), 'Comparison starts')
    #     deblurer = Debluring()
    #     upscaler = SuperResolution()
    #
    #     for input_file_idx, input_file in enumerate(NAFNetTest.input_files):
    #         img = cv2.imread(str(input_file))
    #         assert img is not None, f'Cannot open {input_file}'
    #         h_resized, w_resized = (s * NAFNetTest.scaling_rate for s in img.shape[:2])
    #
    #         # by debluring model
    #         cv2_resized = cv2.resize(img, (w_resized, h_resized))
    #         assert cv2_resized is not None, f'OpenCV resizing fails for {input_file}'
    #         start = time.time()
    #         deblured = deblurer.deblur(cv2_resized)
    #         assert deblured is not None, f'Debluring fails for {input_file}'
    #
    #         # by super-resolution model
    #         upscaled = upscaler.upscale(img)
    #         assert upscaled is not None, f'Upscaling fails for {input_file}'
    #         elapsed += time.time() - start
    #
    #         # images for verification
    #         outcome_imgs = (img, cv2_resized, deblured, upscaled)
    #         review_img = np.zeros((h_resized, w_resized * len(outcome_imgs), 3))
    #         left = 0
    #         for outcome_img in outcome_imgs:
    #             h, w = outcome_img.shape[:2]
    #             review_img[0:h, left:left + w] = outcome_img
    #             left += w_resized
    #         review_file = NAFNetTest.output_dir.joinpath(input_file.name)
    #         written = cv2.imwrite(str(review_file), review_img)
    #         assert written, f'Cannot write {review_file}'
    #
    #         if time.time() - last_reported >= NAFNetTest.reporting_cycle:
    #             print(datetime.datetime.now(), f'{input_file_idx}/{NAFNetTest.num_input_files} '
    #                                            f'({input_file_idx / NAFNetTest.num_input_files * 100:.2f}%)')
    #             last_reported = time.time()
    #
    #     self.monitor.stop()
    #     print('Elapsed=', elapsed, 'Elapsed per file=', elapsed / NAFNetTest.num_input_files)
    #     print(datetime.datetime.now(), 'Conversion finished')


if __name__ == '__main__':
    import unittest
    suite = unittest.TestLoader().loadTestsFromTestCase(NAFNetTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
