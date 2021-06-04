import os

import gc
from tensorflow.python.keras import regularizers

from metric_logging import log_scalar
import tensorflow


from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, MaxPool2D, Softmax, Concatenate, Flatten, GlobalAveragePooling2D

from tensorflow.keras.models import Model

from tensorflow import keras

from envs import Sokoban

import numpy as np

from supervised import DataCreatorSokobanPixelDiff


# def perfect_sequence(y_true, y_pred):
#     return keras.backend.mean(
#         keras.backend.min(
#             keras.backend.cast(
#                 keras.backend.equal(
#                     keras.backend.argmax(y_true, axis=-1),
#                     keras.backend.argmax(y_pred, axis=-1)
#                 ),
#                 'float32'
#             ),
#             axis=-1
#         )
#     )
#
#
# def accuracy_ignore_class(y_true_ohe, y_pred_logits, class_to_ignore):
#     y_true_class = keras.backend.argmax(y_true_ohe)
#     y_pred_class = keras.backend.argmax(y_pred_logits)
#     not_ignored = keras.backend.cast(
#         keras.backend.not_equal(y_pred_class, class_to_ignore), 'int32'
#     )
#     matches = keras.backend.cast(
#         keras.backend.equal(y_true_class, y_pred_class), 'int32'
#     ) * not_ignored
#     accuracy = (
#             keras.backend.sum(matches) /
#             keras.backend.maximum(keras.backend.sum(not_ignored), 1)
#     )
#     return accuracy
#
#
# def get_function_accuracy_ignore_padding(padding_token):
#     def accuracy_ignore_padding(x, y):
#         return accuracy_ignore_class(x, y, padding_token)
#     return accuracy_ignore_padding


class GoalPredictorPixelDiff:
    def __init__(
        self,
        num_layers=5,
        batch_norm=True,
        model_id=None,
        learning_rate=0.01,
        kernel_size=(5, 5),
        weight_decay=0.
    ):

        self.core_env = Sokoban()
        self.dim_room = self.core_env.get_dim_room()

        self.num_layers = num_layers
        self.batch_norm = batch_norm
        self.model_id = model_id

        self._model = None
        self._predictions_counter = 0
        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.weight_decay = weight_decay

    def construct_networks(self):
        if self._model is None:
            if self.model_id is None:
                input_state = Input(batch_shape=(None, None, None, 7))
                input_condition = Input(batch_shape=(None, None, None, 7))

                layer = Concatenate()([input_state, input_condition])

                for _ in range(self.num_layers):
                    layer = Conv2D(
                        filters=64, kernel_size=self.kernel_size,
                        padding='same', activation='relu',
                        kernel_regularizer=regularizers.l2(self.weight_decay),
                                   )(layer)
                    if self.batch_norm:
                        layer = BatchNormalization()(layer)

                branch1 = Dense(
                    7, activation='relu',
                    kernel_regularizer=regularizers.l2(self.weight_decay)
                )(layer)
                branch1 = Flatten()(branch1)

                branch2 = Dense(
                    1, activation='relu',
                    kernel_regularizer=regularizers.l2(self.weight_decay)
                )(layer)
                # output2 = Flatten()(output2)
                branch2 = GlobalAveragePooling2D()(branch2)


                output = Concatenate()([branch1, branch2])
                output = Softmax()(output)

                self._model = Model(
                    inputs=[input_state, input_condition],
                    outputs=output
                )
                self._model.compile(
                    loss='categorical_crossentropy',
                    metrics='accuracy',
                    optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
                )

                self.data_creator = DataCreatorSokobanPixelDiff()
            else:
                self.load_model(self.model_id)

    def reset_predictions_counter(self):
        self._predictions_counter = 0

    def read_predictions(self):
        return self._predictions_counter

    def load_data(self, dataset_file):
        self.data_creator.load(dataset_file)

    def fit_and_dump(self, x, y, validation_data, epochs, dump_folder,
                     checkpoints=None):

        # try:
        #     print(f'making directory {dump_folder}')
        #     os.mkdir(dump_folder)
        # except:
        #     print(f'cannot make directory {dump_folder}')

        for epoch in range(epochs):
            history = self._model.fit(x, y, epochs=1, validation_data=validation_data)
            train_history = history.history
            for metric, value in train_history.items():
                log_scalar(metric, epoch, value[0])
            if checkpoints is not None and epoch in checkpoints:
                print(f'saving model after {epoch} epochs.')
                self.save_model(os.path.join(dump_folder, f'epoch_{epoch}'))
            gc.collect()
        self.save_model(os.path.join(dump_folder, f'epoch_{epoch}'))

    def predict_pdf(self, input, condition):
        self._predictions_counter += 1
        raw =  self._model.predict([np.array([input]), np.array([condition])])[0]
        return raw

    def predict_pdf_batch(self, input_boards, conditions):
        self._predictions_counter += 1
        raw =  self._model.predict([input_boards, conditions])
        return raw

    def save_model(self, model_id):
        self._model.save(model_id)

    def load_model(self, model_id):
        self._model = tensorflow.keras.models.load_model(model_id)

    def flat_to_2d(self, n):
        element = n % 7
        base_n = n // 7
        x = base_n // self.dim_room[0]
        y = base_n % self.dim_room[1]
        return x, y, element

    def smart_sample(self, pdf, internal_confidence_level):
        assert internal_confidence_level > 0 and internal_confidence_level < 1, 'confidence_level must be between 0 and 1'
        out = []
        out_p = []
        for idx in reversed(np.argsort(pdf)):
            out.append(self.flat_to_2d(idx))
            out_p.append(pdf[idx])
            if sum(out_p) > internal_confidence_level:
                break
        return out, out_p
