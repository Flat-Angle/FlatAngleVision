import tensorflow._api.v2.compat.v1 as tf
Flags = tf.app.flags

#choice = input("Press 't' to train; 'p' to predict:\n")
choice = 'p'
if choice == 'p':

    # 输出图片保存文件路径（需修改，最后的斜杠一定要有）
    Flags.DEFINE_string('output_dir', 'D:/PyCharm 2021.1.3/PycharmProjects/flatangleAPI/static/', 'The output directory of the checkpoint')

    Flags.DEFINE_string('summary_dir', './Enhancement/result/log/', 'The dirctory to output the summary')
    Flags.DEFINE_string('mode', 'test', 'The mode of the model train, test.')
    Flags.DEFINE_string('checkpoint', './Enhancement/SRGAN_pre-trained/model-200000', 'If provided, the weight will be restored from the provided checkpoint')
    Flags.DEFINE_boolean('pre_trained_model', True, 'If set True, the weight will be loaded but the global_step will still '
                                                    'be 0. If set False, you are going to continue the training. That is, '
                                                    'the global_step will be initiallized from the checkpoint, too')
    Flags.DEFINE_string('pre_trained_model_type', 'SRResnet', 'The type of pretrained model (SRGAN or SRResnet)')
    Flags.DEFINE_boolean('is_training', False, 'Training => True, Testing => False')
    Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')
    Flags.DEFINE_string('task', 'SRGAN', 'The task: SRGAN, SRResnet')
    # The data preparing operation
    Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch')

    # 需要处理的图片的文件路径（需修改，最后的斜杠一定要有）
    Flags.DEFINE_string('input_dir_LR', 'D:/PyCharm 2021.1.3/PycharmProjects/flatangleAPI/before/', 'The directory of the input resolution input data')

    Flags.DEFINE_string('input_dir_HR', './Enhancement/data/test_HR/', 'The directory of the high resolution input data')
    Flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
    Flags.DEFINE_boolean('random_crop', True, 'Whether perform the random crop')
    Flags.DEFINE_integer('crop_size', 24, 'The crop size of the training image')
    Flags.DEFINE_integer('name_queue_capacity', 2048, 'The capacity of the filename queue (suggest large to ensure'
                                                      'enough random shuffle.')
    Flags.DEFINE_integer('image_queue_capacity', 2048, 'The capacity of the image queue (suggest large to ensure'
                                                       'enough random shuffle')
    Flags.DEFINE_integer('queue_thread', 10, 'The threads of the queue (More threads can speedup the training process.')
    # Generator configuration
    Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')
    # The content loss parameter
    Flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature used in perceptual loss')
    Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
    Flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss')
    Flags.DEFINE_float('vgg_scaling', 0.0061, 'The scaling factor for the perceptual loss if using vgg perceptual loss')
    # The training parameters
    Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
    Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
    Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
    Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
    Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
    Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
    Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
    Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
    Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
    Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')
    print(" Configure Settings for predicting ")

if choice == 't':

    Flags.DEFINE_string('output_dir', '/experiment_SRGAN_VGG54/', 'The output directory of the checkpoint')
    Flags.DEFINE_string('summary_dir', '/experiment_SRGAN_VGG54/log/', 'The dirctory to output the summary')
    Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
    Flags.DEFINE_string('checkpoint', './experiment_SRGAN_MSE/model-500000',
                        'If provided, the weight will be restored from the provided checkpoint')
    Flags.DEFINE_boolean('pre_trained_model', True,
                         'If set True, the weight will be loaded but the global_step will still '
                         'be 0. If set False, you are going to continue the training. That is, '
                         'the global_step will be initiallized from the checkpoint, too')
    Flags.DEFINE_string('pre_trained_model_type', 'SRGAN', 'The type of pretrained model (SRGAN or SRResnet)')
    Flags.DEFINE_boolean('is_training', True, 'Training => True, Testing => False')
    Flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')
    Flags.DEFINE_string('task', 'SRGAN', 'The task: SRGAN, SRResnet')
    # The data preparing operation
    Flags.DEFINE_integer('batch_size', 16, 'Batch size of the input batch')
    Flags.DEFINE_string('input_dir_LR', './data/RAISE_LR/', 'The directory of the input resolution input data')
    Flags.DEFINE_string('input_dir_HR', './data/RAISE_HR/', 'The directory of the high resolution input data')
    Flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
    Flags.DEFINE_boolean('random_crop', True, 'Whether perform the random crop')
    Flags.DEFINE_integer('crop_size', 24, 'The crop size of the training image')
    Flags.DEFINE_integer('name_queue_capacity', 4096, 'The capacity of the filename queue (suggest large to ensure'
                                                      'enough random shuffle.')
    Flags.DEFINE_integer('image_queue_capacity', 4096, 'The capacity of the image queue (suggest large to ensure'
                                                       'enough random shuffle')
    Flags.DEFINE_integer('queue_thread', 10, 'The threads of the queue (More threads can speedup the training process.')
    # Generator configuration
    Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')
    # The content loss parameter
    Flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature used in perceptual loss')
    Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
    Flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss')
    Flags.DEFINE_float('vgg_scaling', 0.0061, 'The scaling factor for the perceptual loss if using vgg perceptual loss')
    # The training parameters
    Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
    Flags.DEFINE_integer('decay_step', 100000, 'The steps needed to decay the learning rate')
    Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
    Flags.DEFINE_boolean('stair', True, 'Whether perform staircase decay. True => decay in discrete interval.')
    Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
    Flags.DEFINE_integer('max_epoch', 10, 'The max epoch for the training')
    Flags.DEFINE_integer('max_iter', 200000, 'The max iteration of the training')
    Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
    Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
    Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')
    print(" Configure Settings for training ")

FLAGS = Flags.FLAGS