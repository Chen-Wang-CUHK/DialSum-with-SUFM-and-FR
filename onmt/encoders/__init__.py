"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
# from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.rnn_encoder_coref import RNNEncoderCoref, SeqHRECoref
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder

# changed by wchen: RNNEncoder is replaced with RNNEncoderCoref
str2enc = {"rnn": RNNEncoderCoref, "brnn": RNNEncoderCoref, "seq_hre_brnn": SeqHRECoref, "cnn": CNNEncoder,
           "transformer": TransformerEncoder, "img": ImageEncoder,
           "audio": AudioEncoder, "mean": MeanEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoderCoref", "SeqHRECoref", "CNNEncoder",
           "MeanEncoder", "str2enc"]
