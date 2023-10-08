#!/usr/bin/env python
# coding: utf-8

# # BHARAT INTERNSHIP
# 
# # NAME-VALLELA MANASA
# 
# # TASK-3 NUMBER RECOGNITION 

# # Importing libraries

# In[19]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


# # Load and preprocess the MNIST dataset

# In[20]:


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# # Normalize pixel values to be between 0 and 1

# In[21]:


train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255


# # One-hot encode the labels

# In[22]:


train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# # Build the neural network model

# In[23]:


model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 images
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 output classes (digits 0-9)


# # Compile the model

# In[24]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# # Train the model

# In[25]:


model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)


# # Evaluate the model on the test data

# In[26]:


index_to_predict = 0
prediction = model.predict(test_images[index_to_predict:index_to_predict + 1])
predicted_label = np.argmax(prediction)
print(f"Predicted label: {predicted_label}")


# In[27]:


test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


# # Display the test image and its true label

# In[28]:


plt.imshow(test_images[index_to_predict], cmap='gray')
plt.title(f"True label: {np.argmax(test_labels[index_to_predict])}")
plt.show()


# In[ ]:




