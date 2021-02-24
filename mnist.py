import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2 as cv
import numpy as np


# Create neural network model.
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


# 28 x 28 pixel images of hand-written digits (0 - 9).
mnist = tf.keras.datasets.mnist

# Unpack data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Visualizing data.
plt.imshow(x_train[0], cmap='binary')
plt.title('Example visualization of test data')
plt.show()

# Create first model.
nn_model = create_model()

# Create checkpoint path and directory for storing saved model weights.
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback.
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

# Train model.
nn_model.fit(x_train, y_train, epochs=3, callbacks=[cp_callback])

# Define loss and accuracy.
val_loss, val_acc = nn_model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# Create a new model for loading previously saved weights.
new_model = create_model()

# Check untrained model accuracy.
# Should be very low as weights are randomized.
val_loss, val_acc = new_model.evaluate(x_test, y_test)
print('Untrained model accuracy: {:5.2f}%'.format(100*val_acc))


# Load the saved weights from previously trained model.
new_model.load_weights(checkpoint_path)

# Check restored model accuracy with weights from previously trained model.
# Should be the same as the trained model's test accuracy.
val_loss, val_acc = new_model.evaluate(x_test, y_test)
print('Restored model accuracy: {:5.2f}%'.format(100*val_acc))

# Tracks number of correct predictions
correct_guesses = 0

for i in range(0, 10):
    img = cv.imread(f'{i}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = np.argmax(nn_model.predict(img))
    print(f'The prediction is: {prediction}')
    if prediction == i:
        print('Correct!')
        correct_guesses = correct_guesses + 1
    else:
        print('Boo! Incorrect!!')
    plt.imshow(img[0], cmap='binary')
    plt.show()

# Prints accuracy of neural network using custom test data
print(f'The prediction rate was {correct_guesses * 10}% correct!')
