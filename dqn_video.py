import numpy as np
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from collections import deque
import random
import matplotlib.pyplot as plt
import cv2

def build_model(input_shape, action_space):
    input_layer = Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (8, 8), strides=(4, 4), activation="relu")(input_layer)
    conv2 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation="relu")(conv1)
    conv3 = layers.Conv2D(64, (3, 3), activation="relu")(conv2)
    flatten = layers.Flatten()(conv3)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    output = layers.Dense(action_space, activation="linear")(dense1)

    model = keras.Model(inputs=input_layer, outputs=output)
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.00025))
    return model

# DQN Ajanı tanımı
class DQNAgent:
    def __init__(self, input_shape, action_space):
        self.input_shape = input_shape
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.model = build_model(input_shape, action_space)
        self.target_model = build_model(input_shape, action_space)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Video karelerini işleme
def preprocess_frame(frame):
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    return frame

def stack_frames(frames, frame):
    frames.append(frame)
    if len(frames) < 4:
        while len(frames) < 4:  # Eğer 4 kareden azsa boş karelerle doldur
            frames.append(np.zeros_like(frame))
    stacked_frames = np.concatenate(frames, axis=-1)  # Çerçeveleri son eksende birleştir
    return stacked_frames


if __name__ == "__main__":
    video_path = "visual/video.mp4"  # Video dosyasının yolu
    cap = cv2.VideoCapture(video_path)
    input_shape = (84, 84, 12)  # Giriş şekli (84x84 piksellik 12 kanal)
    action_space = 3  # Eylem uzayı (örneğin, sol, sağ, dur)
    agent = DQNAgent(input_shape, action_space)
    episodes = 32  # Eğitim bölümleri sayısı
    batch_size = 32  # Mini grup boyutu

    rewards = [] # Toplam ödülleri tutmak için boş liste

    for e in range(episodes):
        frames = deque(maxlen=4)  # Çerçeve belleği
        ret, frame = cap.read()  # Videodan bir kare oku
        if not ret:
            break  # Video biterse döngüden çık
        frame = preprocess_frame(frame)  # Kareyi işle
        state = stack_frames(frames, frame)  # Çerçeveleri istifle
        state = np.expand_dims(state, axis=0)  # Batch boyutunu ekle (shape: (1, 84, 84, 12))

        total_reward = 0  # Bölüm başında toplam ödülü sıfırla

        for time in range(16):
            action = agent.act(state)  # Ajanın eylemini seç
            ret, next_frame = cap.read()  # Sonraki kareyi oku
            if not ret:
                break  # Video biterse döngüden çık
            next_frame = preprocess_frame(next_frame)  # Sonraki kareyi işle
            next_state = stack_frames(frames, next_frame)  # Çerçeveleri istifle
            next_state = np.expand_dims(next_state, axis=0)  # Batch boyutunu ekle (shape: (1, 84, 84, 12))
            reward = 1  # Ödül fonksiyonu (daha karmaşık olabilir)
            total_reward += reward  # Toplam ödülü güncelle
            done = False  # Oyunun bitip bitmediği (video biterse done=True olabilir)
            agent.remember(state, action, reward, next_state, done)
            state = next_state  # Mevcut durumu güncelle
            if done:
                agent.update_target_model()  # Hedef modeli güncelle
                print(f"Episode: {e}/{episodes}, Time: {time}, Epsilon: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)  # Modeli yeniden oynat ve eğit

        rewards.append(total_reward)  # Bölümün sonunda toplam ödülü listeye ekle


    cap.release()  # Video dosyasını kapat
    cv2.destroyAllWindows()

    plt.plot(rewards)  # Ödülleri çizmek için grafiği oluştur
    plt.xlabel('Episode')  # X eksenini bölüm numarası olarak etiketle
    plt.ylabel('Total Reward')  # Y eksenini toplam ödül olarak etiketle
    plt.title('Total Reward over Episodes')  # Grafiğin başlığını belirle
    plt.show()  # Grafiği göster