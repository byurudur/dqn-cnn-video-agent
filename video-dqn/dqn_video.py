import matplotlib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
matplotlib.use('TkAgg') # İnteraktifliği kapat
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from collections import deque
import random
import cv2

# Modeli inşa etme fonksiyonu
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
# EPSİLON 0999 - 0990
class DQNAgent:
    def __init__(self, input_shape, action_space):
        self.input_shape = input_shape
        self.action_space = action_space
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.epsilon_decay_min = 0.01
        self.model = build_model(input_shape, action_space)
        self.target_model = build_model(input_shape, action_space)
        self.update_target_model()


        # Başarı barajı
        self.success_threshold = 2000

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
                target[0][action] = reward  # Ceza verildiği durum
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

# Video karelerini işleme fonksiyonu
def preprocess_frame(frame):
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    return frame

# Çerçeveleri istifleme
def stack_frames(frames, frame):
    frames.append(frame)
    if len(frames) < 4:
        while len(frames) < 4:
            frames.append(np.zeros_like(frame))
    stacked_frames = np.concatenate(frames, axis=-1)
    return stacked_frames

# Oyuncunun ölme durumunu kontrol etme fonksiyonu
def check_player_death(frame):
    def check_player_death(frame):
        # Çerçevenin boyutlarını al
        height, width, _ = frame.shape

        # 1. Oyuncunun çerçevede olup olmadığını kontrol
        player_position = np.mean(frame[int(height * 0.8):, int(width * 0.4):int(width * 0.6)])

        if player_position < 0.2:  # Oyuncu çerçevenin dışında olabilir
            print("Oyuncu çerçevenin dışında!")
            return True

        # 2. Oyuncunun belirli bir renkteki engellere çarpma durumu (kırmızı)

        lower_bound = np.array([0, 0, 100], dtype=np.uint8)  # BGR formatında düşük kırmızı tonlar
        upper_bound = np.array([50, 50, 255], dtype=np.uint8)  # BGR formatında yüksek kırmızı tonlar

        # Oyuncu alanındaki piksellerin belirli bir renk aralığında olup olmadığını kontrol et
        player_area = frame[int(height * 0.8):, int(width * 0.4):int(width * 0.6)]
        mask = cv2.inRange(player_area, lower_bound, upper_bound)

        if np.any(mask > 0):  # Eğer oyuncu kırmızı renkli bir engelle çarpışmışsa
            print("Oyuncu bir engele çarptı!")
            return True

        # Eğer yukarıdaki koşullar sağlanmıyorsa, oyuncu ölmedi olarak kabul edilir
        return False


if __name__ == "__main__":
    video_path = "visual/video-oyun.mp4"
    cap = cv2.VideoCapture(video_path)
    input_shape = (84, 84, 12)
    action_space = 3  # Sol, sağ, ortala gibi eylemler
    agent = DQNAgent(input_shape, action_space)
    episodes = 100 # 100 Bölüm kare başına
    batch_size = 32 # 32'lik grup

    rewards = []


    for e in range(episodes):
        frames = deque(maxlen=4)
        ret, frame = cap.read()
        if not ret:
            break
        frame = preprocess_frame(frame)
        state = stack_frames(frames, frame)
        state = np.expand_dims(state, axis=0)

        total_reward = 0
        bayrak = 0  # Bayrak sıfır for döngüsüne giriş
        for time in range(200):

            action = agent.act(state)
            ret, next_frame = cap.read()
            bayrak = bayrak + 1 # Hareket başına bayrak 1 artsın.
            print("bayrak = " + str(bayrak))
            if not ret:
                break
            next_frame = preprocess_frame(next_frame)
            next_state = stack_frames(frames, next_frame)
            next_state = np.expand_dims(next_state, axis=0)

            # Ödül/ceza sistemi

            # Reward/Penalty system
            reward = 0
            done = False

            # Base rewards based on action
            if action == 2:  # Centering
                reward += 1  # Base reward for centering

            if action in [0, 1]:  # Left or right movement
                reward += 0.5  # Movement reward

            # Check if player died
            oyuncu_olur = check_player_death(next_frame)
            if not oyuncu_olur:
                # Reward for avoiding obstacles
                reward += 2  # Additional reward for successfully avoiding an obstacle or enemy

                # Extra reward for each new enemy joined to the team (dummy variable example)
                new_enemy_joined_count = 1  # You may need to implement this logic based on your game environment
                if new_enemy_joined_count > 0:
                    reward += new_enemy_joined_count * 2  # Extra reward for each new enemy that joins the team

            else:
                # Penalty for collision or dying
                reward -= 10  # Large penalty
                done = True  # End episode if player dies

            # Additional Penalty if the agent moves left or right and risks colliding with an obstacle
            if action in [0, 1] and oyuncu_olur:
                reward -= 1  # Small penalty

            # Add further logic if needed for more specific scenarios, such as bonuses for covering a certain distance or penalties for unnecessary movements

            # reward = 0
            # done = False
            #
            # # Ödül/ceza sistemini aksiyona göre düzenle
            # if action == 2:  # Ortalamak
            #     reward = 1
            # elif action in [0, 1]:  # Sol veya sağ hareket
            #     reward = 0.5
            #
            # # Ceza durumu: Eğer ajan ölürse
            # oyuncu_ölür = check_player_death(next_frame)
            # if oyuncu_ölür:
            #     done = True
            #     reward = -10  # Büyük ceza

            total_reward += reward
            print(reward)
            print(total_reward)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

        if done:
                agent.update_target_model()
                print(f"Episode: {e}/{episodes}, Time: {time}, Epsilon: {agent.epsilon:.2}")
                break
        if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        rewards.append(total_reward)

    cap.release()
    cv2.destroyAllWindows()

    # GRAFİK
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward over Episodes')
    plt.show()
