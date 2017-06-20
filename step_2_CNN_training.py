import keras
from models.CNN import cnn_hiararchical_batchnormalisation, l1_smooth_loss, cnn_sigma
from models.DataLoader import Generator


def main():

    # generator for data loading
    gen = Generator(batch_size=128,
                    filename="./data/OTB100_sigma_%d.hdf5",
                    response_map_shape=[(240, 160), (120, 80), (60, 40), (30, 20), (15, 10)]
                    )

    # construct the model here (pre-defined model)
    model = cnn_hiararchical_batchnormalisation()
    #model.load_weights('./checkpoints/weights.14-0.0047.hdf5')

    print(model.summary())

    def schedule(epoch, decay=0.9):
        return base_lr * decay ** (epoch)

    callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/weights_sigma.{epoch:02d}-{val_loss:.4f}.hdf5',
                                                 verbose=1,
                                                 save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule)]

    base_lr = 1e-3
    optim = keras.optimizers.Adam(lr=base_lr)
    model.compile(optimizer=optim,
                  loss=l1_smooth_loss)

    nb_epoch = 50
    model.fit_generator(generator=gen.generate(True),
                          steps_per_epoch=gen.train_batches,
                          epochs=nb_epoch,
                          verbose=1,
                          callbacks=callbacks,
                          validation_data=gen.generate(False),
                          validation_steps=gen.val_batches,
                          workers=1)

if __name__ == "__main__":
    main()

