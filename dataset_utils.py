import os
import math
import random
import scipy.io
import numpy as np
import tensorflow as tf
from PIL import Image

from rasterization_utils.RealRenderer import GizehRasterizor as RealRenderer


def copy_hparams(hparams):
    """Return a copy of an HParams instance."""
    return tf.contrib.training.HParams(**hparams.values())


class GeneralRawDataLoader(object):
    """gets images in bytes, file, with pointer"""
    def __init__(self,
                 image_path,
                 raster_size,
                 test_dataset):
        self.image_path = image_path
        self.raster_size = raster_size
        self.test_dataset = test_dataset

    def get_test_image(self, random_cursor=True, init_cursor_on_undrawn_pixel=False, init_cursor_num=1):
        input_image_data, image_size_test = self.gen_input_images(self.image_path)
        input_image_data = np.array(input_image_data,
                                    dtype=np.float32)  # (1, image_size, image_size, (3)), [0.0-strokes, 1.0-BG]

        return input_image_data, \
               self.gen_init_cursors(input_image_data, random_cursor, init_cursor_on_undrawn_pixel, init_cursor_num), \
               image_size_test

    def gen_input_images(self, image_path):
        img = Image.open(image_path).convert('RGB')
        height, width = img.height, img.width
        max_dim = max(height, width)
        # img to array of bytes
        img = np.array(img, dtype=np.uint8)

        if height != width:
            # Padding to a square image
            if self.test_dataset == 'clean_line_drawings':
                pad_value = [255, 255, 255]
            elif self.test_dataset == 'faces':
                pad_value = [0, 0, 0]
            else:
                # TODO: find better padding pixel value
                pad_value = img[height - 10, width - 10]

            img_r, img_g, img_b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            pad_width = max_dim - width
            pad_height = max_dim - height

            pad_img_r = np.pad(img_r, ((0, pad_height), (0, pad_width)), 'constant', constant_values=pad_value[0])
            pad_img_g = np.pad(img_g, ((0, pad_height), (0, pad_width)), 'constant', constant_values=pad_value[1])
            pad_img_b = np.pad(img_b, ((0, pad_height), (0, pad_width)), 'constant', constant_values=pad_value[2])
            image_array = np.stack([pad_img_r, pad_img_g, pad_img_b], axis=-1)
        else:
            image_array = img

        assert image_array.shape[0] == image_array.shape[1]
        img_size = image_array.shape[0]
        image_array = image_array.astype(np.float32)
        # clean line drawings have white padding
        
            image_array = image_array[:, :, 0] / 255.0  # [0.0-stroke, 1.0-BG]
        
        
        image_array = np.expand_dims(image_array, axis=0)
        return image_array, img_size

    def crop_patch(self, image, center, image_size, crop_size):
        x0 = center[0] - crop_size // 2
        x1 = x0 + crop_size
        y0 = center[1] - crop_size // 2
        y1 = y0 + crop_size
        x0 = max(0, min(x0, image_size))
        y0 = max(0, min(y0, image_size))
        x1 = max(0, min(x1, image_size))
        y1 = max(0, min(y1, image_size))
        patch = image[y0:y1, x0:x1]
        return patch

    def gen_init_cursor_single(self, sketch_image, init_cursor_on_undrawn_pixel, misalign_size=3):
        # sketch_image: [0.0-stroke, 1.0-BG]
        image_size = sketch_image.shape[0]
        if np.sum(1.0 - sketch_image) == 0:
            center = np.zeros((2), dtype=np.int32)
            return center
        else:
            while True:
                center = np.random.randint(0, image_size, size=(2))  # (2), in large size
                patch = 1.0 - self.crop_patch(sketch_image, center, image_size, self.raster_size)
                if np.sum(patch) != 0:
                    if not init_cursor_on_undrawn_pixel:
                        return center.astype(np.float32) / float(image_size)  # (2), in size [0.0, 1.0)
                    else:
                        center_patch = 1.0 - self.crop_patch(sketch_image, center, image_size, misalign_size)
                        if np.sum(center_patch) != 0:
                            return center.astype(np.float32) / float(image_size)  # (2), in size [0.0, 1.0)

    def gen_init_cursors(self, sketch_data, random_pos=True, init_cursor_on_undrawn_pixel=False, init_cursor_num=1):
        init_cursor_batch_list = []
        for cursor_i in range(init_cursor_num):
            if random_pos:
                init_cursor_batch = []
                for i in range(len(sketch_data)):
                    sketch_image = sketch_data[i].copy().astype(np.float32)  # [0.0-stroke, 1.0-BG]
                    center = self.gen_init_cursor_single(sketch_image, init_cursor_on_undrawn_pixel)
                    init_cursor_batch.append(center)

                init_cursor_batch = np.stack(init_cursor_batch, axis=0)  # (N, 2)
            else:
                raise Exception('Not finished')
            init_cursor_batch_list.append(init_cursor_batch)

        if init_cursor_num == 1:
            init_cursor_batch = init_cursor_batch_list[0]
            init_cursor_batch = np.expand_dims(init_cursor_batch, axis=1).astype(np.float32)  # (N, 1, 2)
        else:
            init_cursor_batch = np.stack(init_cursor_batch_list, axis=1)  # (N, init_cursor_num, 2)
            init_cursor_batch = np.expand_dims(init_cursor_batch, axis=2).astype(
                np.float32)  # (N, init_cursor_num, 1, 2)

        return init_cursor_batch

def load_dataset_testing(test_data_base_dir, test_dataset, test_img_name, model_params):
    """args: folder1 as arg1/folder2 as arg2/img as arg3"""
    # folder 2 can only be named 1 of 3 options based on available models
    assert test_dataset in ['clean_line_drawings', 'rough_sketches', 'faces']
    img_path = os.path.join(test_data_base_dir, test_dataset, test_img_name)
    print('Loaded {} from {}'.format(img_path, test_dataset))

    # hyperparameter tuning
    eval_model_params = copy_hparams(model_params)
    eval_model_params.use_input_dropout = 0
    eval_model_params.use_recurrent_dropout = 0
    eval_model_params.use_output_dropout = 0
    eval_model_params.batch_size = 1
    eval_model_params.model_mode = 'sample'

    # hyperparameter tuning
    sample_model_params = copy_hparams(eval_model_params)
    sample_model_params.batch_size = 1  # only sample one at a time
    sample_model_params.max_seq_len = 1  # sample one point at a time

    test_set = GeneralRawDataLoader(img_path, eval_model_params.raster_size, test_dataset=test_dataset)

    result = [test_set, eval_model_params, sample_model_params]
    return result
