# Global imports
from typing import Optional, Dict
import logging
from argparse import ArgumentParser
import sys
import os


class PredConfig:
    is_training = False
    is_saving = False
    is_testing = False
    train_steps_per_epoch = 0
    test_steps=0
    data_path = ""
    batch_size = 0
    train_data_path = ""
    word_freq_dict_path = ""
    entire_model_save_path = ""
    model_weights_save_path = ""

    @classmethod
    def arguments_parser(cls) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument("-w2v", "--save_word2v", dest="save_w2v",
                            help="path to save the tokens embeddings file", metavar="FILE", required=False)
        parser.add_argument("-t2v", "--save_target2v", dest="save_t2v",
                            help="path to save the targets embeddings file", metavar="FILE", required=False)
        parser.add_argument("-l", "--load", dest="load_path",
                            help="path to load the model from", metavar="FILE", required=False)
        parser.add_argument('--save_w2v', dest='save_w2v', required=False,
                            help="save word (token) vectors in word2vec format")
        parser.add_argument('--save_t2v', dest='save_t2v', required=False,
                            help="save target vectors in word2vec format")
        parser.add_argument('--export_code_vectors', action='store_true', required=False,
                            help="export code vectors for the given examples")
        parser.add_argument('--release', action='store_true',
                            help='if specified and loading a trained model, release the loaded model for a lower model '
                                 'size.')
        parser.add_argument("-v", "--verbose", dest="verbose_mode", type=int, required=False, default=1,
                            help="verbose mode (should be in {0,1,2}).")
        parser.add_argument("-lp", "--logs-path", dest="logs_path", metavar="FILE", required=False,
                            help="path to store logs into. if not given logs are not saved to file.")
        parser.add_argument('-tb', '--tensorboard', dest='use_tensorboard', action='store_true',
                            help='use tensorboard during training')
        return parser

    def set_defaults(self):
        self.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION = 10
        self.MAX_TO_KEEP = 10

        # model hyper-params
        self.MAX_CONTEXTS = 200
        self.MAX_TOKEN_VOCAB_SIZE = 1301136
        self.MAX_TARGET_VOCAB_SIZE = 261245
        self.MAX_PATH_VOCAB_SIZE = 911417
        self.DEFAULT_EMBEDDINGS_SIZE = 128
        self.TOKEN_EMBEDDINGS_SIZE = self.DEFAULT_EMBEDDINGS_SIZE
        self.PATH_EMBEDDINGS_SIZE = self.DEFAULT_EMBEDDINGS_SIZE
        self.CODE_VECTOR_SIZE = self.context_vector_size
        self.TARGET_EMBEDDINGS_SIZE = self.CODE_VECTOR_SIZE
        self.DROPOUT_KEEP_RATE = 0.75
        self.SEPARATE_OOV_AND_PAD = False

        #
        self.DL_FRAMEWORK = 'tensorflow'


    def load_from_args(self):
        args = self.arguments_parser().parse_args()

        # Automatically filled, do not edit:
        self.MODEL_LOAD_PATH = args.load_path
        self.RELEASE = args.release
        self.EXPORT_CODE_VECTORS = args.export_code_vectors
        self.SAVE_W2V = args.save_w2v
        self.SAVE_T2V = args.save_t2v
        self.VERBOSE_MODE = args.verbose_mode
        self.LOGS_PATH = args.logs_path
        self.USE_TENSORBOARD = args.use_tensorboard

    def load_from_kwargs(self, **kwargs):

        # Automatically filled, do not edit:
        self.MODEL_LOAD_PATH = kwargs.get('load_path', self.MODEL_LOAD_PATH)
        self.RELEASE = kwargs.get('release', self.RELEASE)
        self.EXPORT_CODE_VECTORS = kwargs.get('export_code_vectors', self.EXPORT_CODE_VECTORS)
        self.SAVE_W2V = kwargs.get('save_w2v', self.SAVE_W2V)
        self.SAVE_T2V = kwargs.get('save_t2v', self.SAVE_T2V)
        self.VERBOSE_MODE = kwargs.get('verbose_mode', self.VERBOSE_MODE)
        self.LOGS_PATH = kwargs.get('logs_path', self.LOGS_PATH)
        self.LOGS_PATH = kwargs.get('use_tensorboard', self.LOGS_PATH)

    def __init__(
            self, set_defaults: bool = False, load_from_args: bool = False, load_from_kwargs: bool = False,
            verify: bool = False, **kwargs: Dict[str, str]
    ):
        # Unused
        self.NUM_TRAIN_EPOCHS: int = 0
        self.SAVE_EVERY_EPOCHS: int = 0
        self.TRAIN_BATCH_SIZE: int = 0
        self.TEST_BATCH_SIZE: int = 0
        self.NUM_BATCHES_TO_LOG_PROGRESS: int = 0
        self.NUM_TRAIN_BATCHES_TO_EVALUATE: int = 0
        self.READER_NUM_PARALLEL_BATCHES: int = 0
        self.SHUFFLE_BUFFER_SIZE: int = 0
        self.CSV_BUFFER_SIZE: int = 0


        self.TOP_K_WORDS_CONSIDERED_DURING_PREDICTION: int = 0
        self.MAX_TO_KEEP: int = 0

        # model hyper-params
        self.MAX_CONTEXTS: int = 0
        self.MAX_TOKEN_VOCAB_SIZE: int = 0
        self.MAX_TARGET_VOCAB_SIZE: int = 0
        self.MAX_PATH_VOCAB_SIZE: int = 0
        self.DEFAULT_EMBEDDINGS_SIZE: int = 0
        self.TOKEN_EMBEDDINGS_SIZE: int = 0
        self.PATH_EMBEDDINGS_SIZE: int = 0
        self.CODE_VECTOR_SIZE: int = 0
        self.TARGET_EMBEDDINGS_SIZE: int = 0
        self.DROPOUT_KEEP_RATE: float = 0
        self.SEPARATE_OOV_AND_PAD: bool = False

        # Automatically filled by `args`.
        self.PREDICT: bool = True
        self.MODEL_SAVE_PATH: Optional[str] = None
        self.MODEL_LOAD_PATH: Optional[str] = None
        self.TRAIN_DATA_PATH_PREFIX: Optional[str] = None
        self.TEST_DATA_PATH: Optional[str] = ''
        self.RELEASE: bool = True
        self.EXPORT_CODE_VECTORS: bool = True
        self.SAVE_W2V: Optional[str] = None
        self.SAVE_T2V: Optional[str] = None
        self.VERBOSE_MODE: int = 0
        self.LOGS_PATH: Optional[str] = None
        self.DL_FRAMEWORK: str = 'tensorflow'
        self.USE_TENSORBOARD: bool = False


        self.__logger: Optional[logging.Logger] = None

        if set_defaults:
            self.set_defaults()
        if load_from_args:
            self.load_from_args()
        if load_from_kwargs:
            self.load_from_kwargs(**kwargs)
        if verify:
            self.verify()

    @property
    def context_vector_size(self) -> int:
        # The context vector is actually a concatenation of the embedded
        # source & target vectors and the embedded path vector.
        return self.PATH_EMBEDDINGS_SIZE + 2 * self.TOKEN_EMBEDDINGS_SIZE

    @property
    def is_loading(self) -> bool:
        return bool(self.MODEL_LOAD_PATH)

    @classmethod
    def get_vocabularies_path_from_model_path(cls, model_file_path: str) -> str:
        vocabularies_save_file_name = "dictionaries.bin"
        return '/'.join(model_file_path.split('/')[:-1] + [vocabularies_save_file_name])

    @classmethod
    def get_entire_model_path(cls, model_path: str) -> str:
        return model_path + '__entire-model'

    @classmethod
    def get_model_weights_path(cls, model_path: str) -> str:
        return model_path + '__only-weights'

    @property
    def model_load_dir(self):
        return '/'.join(self.MODEL_LOAD_PATH.split('/')[:-1])

    @property
    def entire_model_load_path(self) -> Optional[str]:
        if not self.is_loading:
            return None
        return self.get_entire_model_path(self.MODEL_LOAD_PATH)

    @property
    def model_weights_load_path(self) -> Optional[str]:
        if not self.is_loading:
            return None
        return self.get_model_weights_path(self.MODEL_LOAD_PATH)

    def verify(self):
        if not os.path.isdir(self.model_load_dir):
            raise ValueError("Model load dir `{model_load_dir}` does not exist.".format(
                model_load_dir=self.model_load_dir))

    def __iter__(self):
        for attr_name in dir(self):
            if attr_name.startswith("__"):
                continue
            try:
                attr_value = getattr(self, attr_name, None)
            except:
                attr_value = None
            if callable(attr_value):
                continue
            yield attr_name, attr_value

    def get_logger(self) -> logging.Logger:
        if self.__logger is None:
            self.__logger = logging.getLogger('code2vec')
            self.__logger.setLevel(logging.INFO)
            self.__logger.handlers = []
            self.__logger.propagate = 0

            formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')

            if self.VERBOSE_MODE >= 1:
                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.INFO)
                ch.setFormatter(formatter)
                self.__logger.addHandler(ch)

            if self.LOGS_PATH:
                fh = logging.FileHandler(self.LOGS_PATH)
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                self.__logger.addHandler(fh)

        return self.__logger

    def log(self, msg):
        self.get_logger().info(msg)
