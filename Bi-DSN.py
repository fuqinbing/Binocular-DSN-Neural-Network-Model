import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os
import cv2
import time
import numpy as np
from lib_vibe import vibe_gray

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False    


def export_to_single_sheet(Sf_L_EC, Sf_R_EC, Sf_U_EC, Sf_D_EC, PathVal):
    # Extract the base name from the path and create an Excel file name
    base_name = os.path.splitext(os.path.basename(PathVal))[0]
    excel_file_name = f"{base_name}_Sf_Outputs.xlsx"
    
    # Combine the arrays into a single DataFrame
    data = {
        "Sf_L_EC": Sf_L_EC,
        "Sf_R_EC": Sf_R_EC,
        "Sf_U_EC": Sf_U_EC,
        "Sf_D_EC": Sf_D_EC
    }
    df = pd.DataFrame(data)
    
    # Write the DataFrame to a single sheet in the Excel file
    df.to_excel(excel_file_name, sheet_name="Sf_Outputs", index=False)

    print(f"Data has been exported to {excel_file_name}")

def export_depth_to_excel(depth, PathVal):
    # Extract the base name from the path and create an Excel file name
    base_name = os.path.splitext(os.path.basename(PathVal))[0]
    excel_file_name = f"{base_name}_depth_Outputs.xlsx"
    
    # Convert the depth array to a DataFrame
    df = pd.DataFrame({"Depth": depth})
    
    # Write the DataFrame to an Excel file
    df.to_excel(excel_file_name, sheet_name="Depth_Output", index=False)

    print(f"Depth data has been exported to {excel_file_name}")

def export_activation_to_excel(activation, PathVal):
    # Extract the base name from the path and create an Excel file name
    base_name = os.path.splitext(os.path.basename(PathVal))[0]
    excel_file_name = f"{base_name}_activation_Outputs.xlsx"
    
    # Combine the arrays into a single DataFrame
    data = {
        "activation": activation,
    }
    df = pd.DataFrame(data)
    
    # Write the DataFrame to a single sheet in the Excel file
    df.to_excel(excel_file_name, sheet_name="ac_Outputs", index=False)

    print(f"Data has been exported to {excel_file_name}")

def export_prediction_to_excel(prediction, PathVal):
    # Extract the base name from the path and create an Excel file name
    base_name = os.path.splitext(os.path.basename(PathVal))[0]
    excel_file_name = f"{base_name}_prediction_Outputs.xlsx"
    
    # Combine the arrays into a single DataFrame
    data = {
        "activation": activation,
    }
    df = pd.DataFrame(data)
    
    # Write the DataFrame to a single sheet in the Excel file
    df.to_excel(excel_file_name, sheet_name="pred_Outputs", index=False)

    print(f"Data has been exported to {excel_file_name}")

def export_to_excel_all(Sf_L_EC, Sf_R_EC, Sf_U_EC, Sf_D_EC,depth,activation,prediction,PathVal):
    # Extract the base name from the path and create an Excel file name
    base_name = os.path.splitext(os.path.basename(PathVal))[0]
    excel_file_name = f"{base_name}_Outputs_ViBe.xlsx"
    
    # Combine the arrays into a single DataFrame
    data = {
        "Sf_L_EC": Sf_L_EC,
        "Sf_R_EC": Sf_R_EC,
        "Sf_U_EC": Sf_U_EC,
        "Sf_D_EC": Sf_D_EC,
        "Activation": activation,
        "Depth": depth,
        "Spike": prediction,
    }
    df = pd.DataFrame(data)
    
    # Write the DataFrame to a single sheet in the Excel file
    df.to_excel(excel_file_name, sheet_name="Outputs", index=False)

    print(f"Data has been exported to {excel_file_name}")




@njit
def compute_Pf_gray(L_f,gray_frame):

    M, N = L_f.shape
    
    P_f = np.zeros((M, N), dtype=L_f.dtype)
    
    for i in range(M):
        for j in range(N):
            if L_f[i, j] > 0:
                P_f[i, j] = gray_frame[i, j]
    
    return P_f


@njit
def compute_Pf(L_f):

    M, N = L_f.shape
    
    P_f = np.zeros((M, N), dtype=L_f.dtype)
    
    for i in range(M):
        for j in range(N):
            if L_f[i, j] > 0:
                P_f[i, j] = L_f[i, j]
    
    return P_f


@njit
def SUM_f(S_layer,t):

    I_W = S_layer.shape[0]

    I_H = S_layer.shape[1]

    SUM_f = np.sum(np.abs(S_layer[:,:,t]))

    S_f = 1 / (1 +  (np.exp((-SUM_f * (1 / (I_W * I_H)))) ))

    return S_f

@njit
def SUM_f_Slayer(P_layer,I_layer,S_layer,t):

    T_rs = 12
    
    I_W = P_layer.shape[0]

    I_H = P_layer.shape[1]
    for x in range(I_W):
        for y in range(I_H):
            S_layer[x, y,t] = (P_layer[x,y,t] - I_layer[x,y,t]) * ((P_layer[x,y,t] - I_layer[x,y,t]) >= T_rs) 

    SUM_f = np.sum(np.abs(S_layer[:,:,t]))

    S_f = 1 / (1 +  (np.exp((-SUM_f * (1 / (I_W * I_H)))) ))


    return S_f

def average_pooling(input_array, scale_factor,t):
    H, W, T = input_array.shape
    new_H, new_W = H // scale_factor, W // scale_factor
    pooled_array = np.zeros((new_H, new_W, T))

    # for t in range(T):
    #     pooled_array[:, :, t] = cv2.resize(
    #         input_array[:, :, t], (new_W, new_H), interpolation=cv2.INTER_AREA
    #     )

    pooled_array[:, :, t] = cv2.resize(
        input_array[:, :, t], (new_W, new_H), interpolation=cv2.INTER_AREA
    )

    return pooled_array[:, :, t]

@njit
def update_summation_layer_U(I_H,I_W,P_layer,I_layer,S_layer,t,inhibition_radius,local_inhibition_weight,global_inhibition_weight):
    # Update summation layer with excitatory and inhibitory inputs

    # excitation = P_layer[:,:,t]
    T_rs = 12 
    SUM_f = 0
    S_f =  0
    epsilon = 1e-10  # Small value to prevent division by zero
    I_L_32x32 = 0

    # for x in range(input_shape[0]):
    #     for y in range(input_shape[1]):
    for x in range(I_W):
        for y in range(I_H):
            # Excitation from E layer
            # excitation = P_layer[x, y]
            
            # Inhibition from I layer to neighboring cells in S layer (within inhibition radius)
            inhibition = 0
            for i in range(1, inhibition_radius + 1):
                # if x + i < input_shape[0]:

                if x + i < I_W:
                    I_layer[x,y,t] += P_layer[x + i, y,t-1] * local_inhibition_weight
            
            I_layer[x,y,t] *= global_inhibition_weight


    return I_layer

@njit
def update_summation_layer_D(I_H,I_W,P_layer,I_layer,S_layer,t,inhibition_radius,local_inhibition_weight,global_inhibition_weight):
    # Update summation layer with excitatory and inhibitory inputs

    # excitation = P_layer[:,:,t]
    T_rs = 12 
    SUM_f = 0
    S_f =  0
    epsilon = 1e-10  # Small value to prevent division by zero

    for x in range(I_W):
        for y in range(I_H):
           
            # Inhibition from I layer to neighboring cells in S layer (within inhibition radius)
            inhibition = 0
            for i in range(1, inhibition_radius + 1):
                # if x + i < input_shape[0]:

                if x - i > 0:
                    I_layer[x,y,t] += P_layer[x - i, y,t-1] * local_inhibition_weight
            
            # Apply global inhibition weight
            # inhibition *= global_inhibition_weight

            I_layer[x,y,t] *= global_inhibition_weight
            
    return I_layer

@njit
def update_summation_layer_R(I_H,I_W,P_layer,I_layer,S_layer,t,inhibition_radius,local_inhibition_weight,global_inhibition_weight):
    # Update summation layer with excitatory and inhibitory inputs

    # excitation = P_layer[:,:,t]
    T_rs = 12 
    SUM_f = 0
    S_f =  0
    epsilon = 1e-10  # Small value to prevent division by zero

    for x in range(I_W):
        for y in range(I_H):
       
            # Inhibition from I layer to neighboring cells in S layer (within inhibition radius)
            inhibition = 0
            for i in range(1, inhibition_radius + 1):

                if y - i > 0:
                    I_layer[x,y,t] += P_layer[x, y - i,t-1] * local_inhibition_weight
            
            # Apply global inhibition weight
            # inhibition *= global_inhibition_weight

            I_layer[x,y,t] *= global_inhibition_weight
            

    return I_layer

@njit
def update_summation_layer_L(I_H,I_W,P_layer,I_layer,S_layer,t,inhibition_radius,local_inhibition_weight,global_inhibition_weight):
    # Update summation layer with excitatory and inhibitory inputs

    # excitation = P_layer[:,:,t]
    T_rs = 12 
    SUM_f = 0
    S_f =  0
    epsilon = 1e-10  # Small value to prevent division by zero

    # for x in range(input_shape[0]):
    #     for y in range(input_shape[1]):
    for x in range(I_W):
        for y in range(I_H):
            # Excitation from E layer
            # excitation = P_layer[x, y]
            
            # Inhibition from I layer to neighboring cells in S layer (within inhibition radius)
            inhibition = 0
            for i in range(1, inhibition_radius + 1):
                # if x + i < input_shape[0]:

                if y + i < I_H:
                    I_layer[x,y,t] += P_layer[x, y + i,t-1] * local_inhibition_weight
            
            # Apply global inhibition weight
            # inhibition *= global_inhibition_weight

            I_layer[x,y,t] *= global_inhibition_weight

    return I_layer


class PresynapticDSN_1:
    def PresynapticDSN(self,PathVal):

        video = cv2.VideoCapture(PathVal)
        FramesCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        I_H = 128 
        I_W = 128 


        X = np.zeros((I_H, I_W, FramesCount))

        M = np.zeros((I_H, I_W, FramesCount))

        D = np.zeros((I_H, I_W, FramesCount))

        P = np.zeros((I_H, I_W, FramesCount))

        P_D = np.zeros((I_H, I_W, FramesCount))


        I_L,I_R,I_U,I_D = np.zeros((I_H, I_W, FramesCount)), \
                              np.zeros((I_H, I_W, FramesCount)), \
                              np.zeros((I_H, I_W, FramesCount)), \
                              np.zeros((I_H, I_W, FramesCount))

        I_L_EC,I_R_EC,I_U_EC,I_D_EC = np.zeros((32,32, FramesCount)), \
                              np.zeros((32, 32, FramesCount)), \
                              np.zeros((32, 32, FramesCount)), \
                              np.zeros((32, 32, FramesCount))


        S_L,S_R,S_U,S_D = np.zeros((I_H, I_W, FramesCount)), \
                              np.zeros((I_H, I_W, FramesCount)), \
                              np.zeros((I_H, I_W, FramesCount)), \
                              np.zeros((I_H, I_W, FramesCount))

        Sf_L_EC,Sf_R_EC,Sf_U_EC,Sf_D_EC = np.zeros((FramesCount)), \
                              np.zeros((FramesCount)), \
                              np.zeros((FramesCount)), \
                              np.zeros((FramesCount))


        S_L_EC,S_R_EC,S_U_EC,S_D_EC = np.zeros((32, 32, FramesCount)), \
                              np.zeros((32, 32, FramesCount)), \
                              np.zeros((32, 32, FramesCount)), \
                              np.zeros((32, 32, FramesCount))


        P_32x32 = np.zeros((32, 32, FramesCount))

        segmentation_map_USE = np.zeros((128, 128, FramesCount)) 

        gray_frame_USE = np.zeros((128, 128, FramesCount)) 

        self.G = np.zeros((128, 128, FramesCount))         

        FFI = np.zeros(FramesCount)
        FFI_X = np.zeros(FramesCount)
        FFI_M = np.zeros(FramesCount)

        frame_index = 0
        segmentation_time = 0
        update_time = 0
        t1 = time.time()


        vibe = vibe_gray()

        for t in range(FramesCount):
            ret, frame = video.read()
            if not ret:
                print(f"Frame {t} could not be read. Exiting loop.")
                # break
                continue

            frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_NEAREST)

            frame = frame.astype(np.float32)

            gray_frame = frame[:,:,0]

            if frame_index % 100 == 0:
                print('Frame number: %d' % frame_index)
                
            if frame_index == 0:
                vibe.AllocInit(gray_frame)
            
            t2 = time.time()
            segmentation_map = vibe.Segmentation(gray_frame)
            t3 = time.time()
            vibe.Update(gray_frame, segmentation_map)
            t4 = time.time()
            segmentation_time += (t3-t2)
            update_time += (t4-t3)
            print('Frame %d, segmentation: %.4f, updating: %.4f' % (frame_index, t3-t2, t4-t3))
            segmentation_map = cv2.medianBlur(segmentation_map, 3)
            

            frame_index += 1
            M[:,:,t] = gray_frame.copy()

            gray_frame = compute_Pf_gray(segmentation_map,gray_frame)

            gray_frame_USE[:,:,t] = gray_frame.copy() 
            
            Y = gray_frame_USE

            X[:, :, t] = Y[:,:,t]

            segmentation_map_USE[:,:,t] = segmentation_map.copy() 
            
            D[:,:,t] = segmentation_map_USE[:,:,t] 
            
       
            if t >= 2:

                sum1 = 0
                sum2 = 0
                sum3 = 0
                FFI[0]=0
                FFI_X[0]=0
                FFI_M[0]=0

                for i in range(I_H):
                    for j in range(I_W):
                        sum1 = sum1 + abs(D[i,j,t-1])
                        sum2 = sum2 + abs(X[i,j,t] - X[i,j,t-1])
                        sum3 = sum3 + abs(D[i,j,t] - D[i,j,t-1])


                FFI[t]=(sum1 / (I_H*I_W))
                FFI_X[t]=(sum2 / (I_H*I_W))
                FFI_M[t]=(sum3 / (I_H*I_W))


                factor_M = 1 / (1 + np.exp(-0.1 * (FFI[t] - 20)))  
                factor_X = 1 - factor_M  
                P[:, :, t] = np.abs(X[:, :, t] - X[:, :, t-1]) * factor_X + np.abs(M[:, :, t] - M[:, :, t-1]) * factor_M  

                P_D[:, :, t] = np.abs(D[:, :, t] - D[:, :, t-1]) 
                print('FFI_X: %f ' % factor_X)
                print('FFI_M: %f ' % factor_M)







                P_32x32[:,:,t] = average_pooling(P, 4,t)

                I_L = update_summation_layer_L(I_H,I_W,P_layer=P,I_layer=I_L,S_layer=S_L,t=t,inhibition_radius=1,local_inhibition_weight=5.5,global_inhibition_weight=1.5)
                I_R = update_summation_layer_R(I_H,I_W,P_layer=P,I_layer=I_R,S_layer=S_R,t=t,inhibition_radius=1,local_inhibition_weight=5.5,global_inhibition_weight=1.5)
                I_U = update_summation_layer_U(I_H,I_W,P_layer=P,I_layer=I_U,S_layer=S_U,t=t,inhibition_radius=1,local_inhibition_weight=5.5,global_inhibition_weight=1.5)
                I_D = update_summation_layer_D(I_H,I_W,P_layer=P,I_layer=I_D,S_layer=S_D,t=t,inhibition_radius=1,local_inhibition_weight=5.5,global_inhibition_weight=1.5)

                I_L_EC[:,:,t] = average_pooling(I_L, 4,t)
                I_R_EC[:,:,t] = average_pooling(I_R, 4,t)
                I_U_EC[:,:,t] = average_pooling(I_U, 4,t)
                I_D_EC[:,:,t] = average_pooling(I_D, 4,t)

                Sf_L_EC[t] = SUM_f_Slayer(P_layer=P_32x32,I_layer=I_L_EC,S_layer=S_L_EC,t=t)
                Sf_R_EC[t] = SUM_f_Slayer(P_layer=P_32x32,I_layer=I_R_EC,S_layer=S_R_EC,t=t)
                Sf_U_EC[t] = SUM_f_Slayer(P_layer=P_32x32,I_layer=I_U_EC,S_layer=S_U_EC,t=t)
                Sf_D_EC[t] = SUM_f_Slayer(P_layer=P_32x32,I_layer=I_D_EC,S_layer=S_D_EC,t=t)


        self.G = P_D.copy()
        video.release()
        cv2.destroyAllWindows()

        curve_color = (0/255, 0/255, 234/255)
        bar_color = (10/255, 5/255, 5/255)
        dashed_line_color = (214/255, 226/255, 228/255)
        dashed_color = (164/255, 166/255, 168/255)

        x = list(range(FramesCount))


        fig, ax = plt.subplots(figsize=(10, 6),dpi=300)
        ax.plot(x, Sf_L_EC, color=curve_color, linestyle='--', linewidth=2.7, label='Sf_L') 
        ax.plot(x, Sf_R_EC, color='black', linestyle='-.', linewidth=2.7, label='Sf_R')     
        ax.plot(x, Sf_U_EC, color='red', linestyle=':', linewidth=2.7, label='Sf_U')        
        ax.plot(x, Sf_D_EC, color='green', linestyle='-', linewidth=2.7, label='Sf_D')      

        ax.set_ylim([0.45, 1.1])


        plt.xticks(fontsize=30) 
        ax.set_ylabel('Membrane Potential', fontsize=32)
        

        ax.legend(loc='upper right',fontsize=10)

        plt.tight_layout()
        plt.show()




        return Sf_L_EC,Sf_R_EC,Sf_U_EC,Sf_D_EC





@njit
def calculate_disparity(G_l, G_r, R, C,T):
    disparity_map = np.zeros((C,T))  
    disparity = np.zeros(T)
    disparity_score = 0

    for t in range(1,T):
        for d in range(0, C):  
            for x in range(R):
                for y in range(C - d):

                    disparity_score += G_l[x, y + d,t] * G_r[x, y,t]

            disparity_map[d,t] = disparity_score  

        disparity[t] = np.argmax(disparity_map[:,t], axis=0)  
    return disparity


# Depth calculation function based on disparity
def calculate_depth(disparity):
    pixel_size = 1
    baseline = 7.5
    focal_length = 441.25




    depth = np.zeros_like(disparity, dtype=np.float32)
    for t in range(disparity.shape[0]):
        if disparity[t] != 0:
            depth[t] = (baseline * focal_length) / (disparity[t] * pixel_size)
        else:

            depth[t] = 0.0  # Set depth to infinity where no disparity

    return depth

def plot_depth(depth):

    depth_m = depth / 100

    plt.figure(figsize=(10, 3),dpi=300)
    plt.plot(depth_m, label='Depth', color='blue')
    plt.title('Depth Variation Based on Disparity')
    plt.xlabel('Pixel Index')
    plt.ylabel('Depth (m)')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_disparity(disparity):

    plt.figure(figsize=(10, 3),dpi=300)
    plt.plot(disparity, label='disparity', color='red')
    plt.title('Disparity')
    plt.xlabel('Pixel Index')
    plt.ylabel('disparity (cm)')
    plt.grid(True)
    plt.legend()
    plt.show()

class DirectionalNeuron:
    def __init__(self, directions=['L', 'R', 'U', 'D']):
        self.directions = directions
        self.inputs = {d: 0 for d in directions}  
        self.inputs['depth'] = 0  
        self.Sf_values = {d: [] for d in directions}
        self.Sf_values['depth'] = []  

    def update_input(self, direction, value):
        if direction in self.inputs:
            self.inputs[direction] = value

    def get_directional_input_array(self):
        return np.array([self.inputs[d] for d in self.directions])

    def get_depth_input(self):
        return self.inputs['depth']

    def update_Sf_values(self, sf_data):
        for key in sf_data:
            if key in self.Sf_values:
                self.Sf_values[key].extend(sf_data[key])

    def get_Sf_values(self):
        return self.Sf_values


class Agent:
    def __init__(self, weights, threshold, input_neurons):
        self.weights = weights
        self.threshold = threshold
        self.fitness = 0
        self.input_neurons = input_neurons
        self.spikes = []
        self.spike_threshold = 5
        self.time_window = 5

        self.directional_hidden_weights = np.reshape(self.weights[:8 * 4], (8, 4))  
        self.depth_hidden_weight = self.weights[8 * 4]  

        self.output_weights = np.reshape(self.weights[8 * 4 + 1:8 * 4 + 1 + 9], (1, 9))

    def mutate(self, mutation_rate=0.2):
        for i in range(len(self.weights)):
            if random.random() < mutation_rate:
                self.weights[i] += np.random.uniform(-0.5, 0.5)

        if random.random() < mutation_rate:
            self.threshold += np.random.uniform(-2.5, 2.5)

        if random.random() < mutation_rate:
            self.weights[32] += np.random.uniform(-0.5, 0.5)



    def crossover(self, other_agent):
        crossover_point = random.randint(1, len(self.weights) - 1)
        child_weights = np.concatenate((self.weights[:crossover_point], other_agent.weights[crossover_point:]))
        child_threshold = (self.threshold + other_agent.threshold) / 2
        return Agent(child_weights, child_threshold, self.input_neurons)

    def propagate(self):
        directional_input_array = self.input_neurons.get_directional_input_array()  
        directional_hidden_activation = np.dot(self.directional_hidden_weights, directional_input_array)
        directional_hidden_activation = np.maximum(0, directional_hidden_activation)

        depth_input = self.input_neurons.get_depth_input()
        depth_hidden_activation = max(0, depth_input * self.depth_hidden_weight)  

        combined_hidden_activation = np.append(directional_hidden_activation, depth_hidden_activation)

        output_activation = np.dot(self.output_weights, combined_hidden_activation)
        output_activation = np.maximum(0, output_activation)  
        
        return output_activation


    def spiking_mechanism(self, activation):
        excitation_level = np.sum(activation)
        spike = 1 if excitation_level >= self.threshold else 0
        self.spikes.append(spike)
        
        if len(self.spikes) > self.time_window:
            self.spikes.pop(0)
        
        return sum(self.spikes) >= self.spike_threshold

    def evaluate(self, events):
        total_score = sum(4 if e['is_collision'] else 1 for e in events)
        score = 0
        score_events = []

        for event in events:
            sf_file = f"{os.path.splitext(os.path.basename(event['Path']))[0]}_Sf_Outputs.xlsx"
            depth_file = f"{os.path.splitext(os.path.basename(event['Path']))[0]}_depth_Outputs.xlsx"
            try:
                sf_data = pd.read_excel(sf_file)
                sf_values = {
                    'L': sf_data['Sf_L_EC'].values.flatten(),
                    'R': sf_data['Sf_R_EC'].values.flatten(),
                    'U': sf_data['Sf_U_EC'].values.flatten(),
                    'D': sf_data['Sf_D_EC'].values.flatten()
                }
                self.input_neurons.update_Sf_values(sf_values)
                
                depth_data = pd.read_excel(depth_file)

                self.input_neurons.update_Sf_values({'depth': depth_data['Depth'].values.flatten()})
                
            except FileNotFoundError:
                print(f"文件 {sf_file} 或 {depth_file} 未找到。")
                continue
            except KeyError as e:
                print(f"文件 {sf_file} 或 {depth_file} 中缺少列 {e}。")
                continue

            video = cv2.VideoCapture(event['Path'])
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            collision_frame = event['Collision_frame']


            recent_predictions = []

            N = 0

            activation_levels = []

            for frame_idx in range(total_frames):
                ret, frame = video.read()
                if not ret:
                    break

                if frame_idx < len(sf_values['L']):
                    for direction in self.input_neurons.directions:
                        self.input_neurons.update_input(direction, sf_values[direction][frame_idx])
                    self.input_neurons.update_input('depth', self.input_neurons.Sf_values['depth'][frame_idx])

                activation = self.propagate()
                prediction = self.spiking_mechanism(activation)

                recent_predictions.append(prediction)

                activation_levels.append(activation.item())  

                if N == 0:

                    if event['is_collision'] :

                        if frame_idx == collision_frame-3:
                            collision_predicted = any(recent_predictions[collision_frame - 30:collision_frame - 3])

                            if not collision_predicted:
                                score += 4
                                N += 1

                                score_events.append(f"碰撞事件 {event['Path']} 在第 {frame_idx} 帧得分增加-1")

                    elif not event['is_collision'] and prediction:
                        score += 1
                        N += 1
                        score_events.append(f"非碰撞事件 {event['Path']} 在第 {frame_idx} 帧得分增加")

            video.release()

            plt.figure()
            plt.plot(range(len(activation_levels)), activation_levels, label='Activation Level')
            
            plt.show()

        self.fitness = (1 - score / total_score)
        self.score_events = score_events

def plot_activation_levels_for_best_agent(agent,Path,Sf_L_EC,Sf_R_EC,Sf_U_EC,Sf_D_EC,depth):
    sf_values = {
        'L': Sf_L_EC,
        'R': Sf_R_EC,
        'U': Sf_U_EC,
        'D': Sf_D_EC
    }


    agent.input_neurons.update_Sf_values(sf_values)
    
    agent.input_neurons.update_Sf_values({'depth': depth})
        
    video = cv2.VideoCapture(Path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


    recent_predictions = []

    N = 0


    activation_levels = []  

    recent_predictions_frame = []

    for frame_idx in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break

        if frame_idx < len(sf_values['L']):
            for direction in agent.input_neurons.directions:
                agent.input_neurons.update_input(direction, sf_values[direction][frame_idx])
            agent.input_neurons.update_input('depth', agent.input_neurons.Sf_values['depth'][frame_idx])

        activation = agent.propagate()
        prediction = agent.spiking_mechanism(activation)

        recent_predictions.append(prediction)

        activation_levels.append(activation.item())  

        if prediction:
            recent_predictions_frame.append(frame_idx)



    video.release()

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Plot the activation levels
    ax.plot(range(len(activation_levels)), activation_levels, label='Activation Level')

    # Add vertical lines for recent prediction frames
    for frame in recent_predictions_frame:
        ax.axvline(x=frame, color='red', linestyle='--')

    # Add labels and title
    ax.set_xlabel('帧')
    ax.set_ylabel('激活值')
    ax.set_title(f"事件的激活值变化: {os.path.basename(Path)}")

    # Show the legend and plot
    ax.legend()
    plt.show()

    return activation_levels,recent_predictions



directional_hidden_weights = np.array([
 [ 1.2396886,  -0.9567193,  -0.9425232,   0.60197985],
 [-0.15928336,  0.03692009,  0.105004,   -1.16635103],
 [-0.21925644,  0.04318147, -0.1674393,   0.13314312],
 [-1.67534786, -0.49209833 ,-0.0228246,  -0.19639414],
 [ 0.73342683 ,-0.42928141 ,-0.66218782,  0.63987775],
 [ 1.18554259 ,-0.82123325, -1.50923107,  0.10795792],
 [ 0.68176564 , 1.14640183 , 0.72152299 ,-0.42079914],
 [ 1.74869872 ,-0.95970035 ,-0.78380663 ,-1.39841214]
])
depth_hidden_weight = -0.7098942363539607
output_weights = np.array([
    [1.48437014,  0.79316137,  0.79687697, -0.64205691, -0.7018617,   1.58827358, 1.43760328, -1.21498595, -0.53566672]
])
threshold = 1.76 



weights = np.concatenate([
    directional_hidden_weights.flatten(),
    [depth_hidden_weight],
    output_weights.flatten()
])

input_neurons = DirectionalNeuron()

best_agent = Agent(weights=weights, threshold=threshold, input_neurons=input_neurons)

best_agent.score_events = [

    {'is_collision': False, 'Path': r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output13_left.mp4", 'Collision_frame': None},
    {'is_collision': False, 'Path': r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output14_left.mp4", 'Collision_frame': None}

]



def main(PathVal_l,PathVal_r):

    DSN_l = PresynapticDSN_1()

    Sf_L_EC_l,Sf_R_EC_l,Sf_U_EC_l,Sf_D_EC_l = DSN_l.PresynapticDSN(PathVal_l)
    G_l = DSN_l.G.copy()

    DSN_R = PresynapticDSN_1()

    Sf_L_EC_r,Sf_R_EC_r,Sf_U_EC_r,Sf_D_EC_r = DSN_R.PresynapticDSN(PathVal_r)
    G_r = DSN_R.G.copy()

    T = G_l.shape[2]

    Sf_L_EC,Sf_R_EC,Sf_U_EC,Sf_D_EC = (Sf_L_EC_l+Sf_L_EC_r)/2,(Sf_R_EC_l+Sf_R_EC_r)/2,(Sf_U_EC_l+Sf_U_EC_r)/2,(Sf_D_EC_l+Sf_D_EC_r)/2 

    # export_to_single_sheet(Sf_L_EC, Sf_R_EC, Sf_U_EC, Sf_D_EC, PathVal_l)
    disparity = calculate_disparity(G_l, G_r, 128, 128,T)
    plot_disparity(disparity)

    depth = calculate_depth(disparity)
    plot_depth(depth)


    activation_levels,prediction = plot_activation_levels_for_best_agent(best_agent,PathVal_l,Sf_L_EC,Sf_R_EC,Sf_U_EC,Sf_D_EC,depth)


    export_to_excel_all(Sf_L_EC, Sf_R_EC, Sf_U_EC, Sf_D_EC,depth,activation_levels,prediction,PathVal_l)


# PathVal = r"E:\\CSDIY\\F_discussion\\LGMD_video\\实验视频\\实验视频\\vehicle stimuli\\c05.mp4"

# PathVal = r"E:\\CSDIY\\F_discussion\\LGMD_video\\实验视频\\实验视频\\特殊刺激\\down-0-255.avi"

# PathVal = r"E:\\CSDIY\\F_discussion\\LGMD_video\\实验视频\\实验视频\\特殊刺激\\output(19).avi"

# PathVal = r"E:\\CSDIY\\F_discussion\\LGMD2\\output.avi"

# PathVal = r"E:\\CSDIY\\F_discussion\\LGMD_video\\a-255-0.avi"

# PathVal = r"E:\\CSDIY\\F_discussion\\LGMD_video\\a-0-255.avi"

# PathVal = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\color_平移_窗帘_近.mp4"

# PathVal = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\color_平移_窗帘_远.mp4"

# PathVal = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\color_正面碰撞_前门.mp4"

# PathVal = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\color_正面碰撞_胶带.mp4"

# PathVal = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\color_正面碰撞_窗帘.mp4"

# PathVal = r"G:\\IJCNN\\Bi-Video-Real\\mono1_平移_窗帘_近.mp4"

# PathVal = r"G:\\IJCNN\\Bi-Video-Real\\mono1_平移_窗帘_远.mp4"

# PathVal = r"G:\\IJCNN\\Bi-Video-Real\\mono2_平移_窗帘_近.mp4"

# PathVal = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\color_左经过_窗帘.mp4"

# PathVal = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono1_左经过_窗帘.mp4"

# PathVal = r"E:\\CSDIY\\F_discussion\\LGMD_video\\实验视频\\实验视频\\synthetic stimuli\\t-180-255.avi"

# PathVal_l = r"G:\\IJCNN\\Bi-Video-Real\\mono1_平移_窗帘_近.mp4"
# PathVal_r = r"G:\\IJCNN\\Bi-Video-Real\\mono2_平移_窗帘_近.mp4"

# PathVal_l = r"G:\\IJCNN\\Bi-Video-Real\\mono1_平移_窗帘_远.mp4"
# PathVal_r = r"G:\\IJCNN\\Bi-Video-Real\\mono2_平移_窗帘_远.mp4"


# PathVal_r = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono2_右经过_窗帘.mp4"
# PathVal_l = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono1_右经过_窗帘.mp4"

# PathVal_r = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono2_左经过_窗帘.mp4"
# PathVal_l = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono1_左经过_窗帘.mp4"

# PathVal_r = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono2_左经过_前门.mp4"
# PathVal_l = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono1_左经过_前门.mp4"

# PathVal_r = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono2_正面碰撞_右_前门.mp4"
# PathVal_l = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono1_正面碰撞_左_前门.mp4"

# PathVal_r = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono2_正面碰撞_窗帘.mp4"
# PathVal_l = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono1_正面碰撞_窗帘.mp4"

# PathVal_r = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono2_正面碰撞_胶带.mp4"
# PathVal_l = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono1_正面碰撞_胶带.mp4"



# PathVal_r = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono2_左经过_前门.mp4"
# PathVal_l = r"E:\\CSDIY\\IJCNN\\Bi-Video-Real\\mono1_左经过_前门.mp4"

PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\looming1_left.mp4"
PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\looming1_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\looming2_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\looming2_right.mp4"


# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\translating1_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\translating1_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\translating2_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\translating2_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\translating3_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\translating3_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\translating4_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\translating4_right.mp4"


# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output2_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output2_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output3_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output3_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output4_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output4_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output5_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output5_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output6_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output6_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output7_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output7_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output8_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output8_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output10_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output10_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output11_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output11_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output12_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output12_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output13_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output13_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output14_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output14_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output15_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output15_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output16_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\Bi-Video\\output16_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-好像碰撞了.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-好像碰撞了.mp4"


# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-师兄1.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-师兄1.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-师兄2.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-师兄2.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-旋转环境.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-旋转环境.mp4"


# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-正面碰撞.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-正面碰撞.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-正面碰撞2.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-正面碰撞2.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-正面碰撞3.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-正面碰撞3.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-正面碰撞4.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-正面碰撞4.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-正面碰撞5.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-正面碰撞5.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-右相机类碰撞.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-右相机类碰撞.mp4"


# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-先慢后快.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-先慢后快.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-斜着平移近.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-斜着平移近.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-斜着平移.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-斜着平移.mp4"


# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-平移近.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-平移近.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-左平移快.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-左平移快.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-左平移较快.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-左平移较快.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-左平移较慢.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-左平移较慢.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-路过2.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-路过2.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-路过1.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-路过1.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-路过5.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-路过5.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-路过6.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-路过6.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-后退.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-后退.mp4"


# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-动态背景4.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-动态背景4.mp4"



# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-黑色小球靠近2.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-黑色小球靠近2.mp4"


# PathVal_l = r"E:\\CSDIY\\F_discussion\\LGMD_video\\实验视频\\实验视频\\ball stimuli\\123.mp4"
# PathVal_r = r"E:\\CSDIY\\F_discussion\\LGMD_video\\实验视频\\实验视频\\ball stimuli\\123.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-Real\\mono1-黑色小球靠近.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-Real\\mono2-黑色小球靠近.mp4"


# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output1mid_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output1mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output2slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output2slow_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output3fast_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output3fast_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output5(5)mid_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output5(5)mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output5(5)slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output5(5)slow_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output5(5)high_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output5(5)high_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output4(15)mid_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output4(15)mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output4(15)slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output4(15)slow_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output4(15)fast_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output4(15)fast_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output6(25)mid_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output6(25)mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output6(25)slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output6(25)slow_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output6(25)fast_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output6(25)fast_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output7(35)mid_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output7(35)mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output7(35)slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output7(35)slow_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output7(35)fast_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output7(35)fast_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output8(45)mid_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output8(45)mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output8(45)slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output8(45)slow_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output8(45)fast_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output8(45)fast_left.mp4"


# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output9(55)mid_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output9(55)mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output9(55)slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output9(55)slow_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output9(55)fast_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-12\\video\\output9(55)fast_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output1(5)fast_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output1(5)fast_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output1(5)mid_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output1(5)mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output1(5)slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output1(5)slow_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output2(15)fast_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output2(15)fast_left.mp4"

PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output2(15)mid_right.mp4"
PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output2(15)mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output2(15)slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output2(15)slow_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output3(25)fast_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output3(25)fast_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output3(25)mid_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output3(25)mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output3(25)slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output3(25)slow_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output4(35)fast_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output4(35)fast_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output4(35)mid_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output4(35)mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output4(35)slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output4(35)slow_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output5(45)fast_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output5(45)fast_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output5(45)mid_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output5(45)mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output5(45)slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output5(45)slow_left.mp4"


# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output6(55)fast_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output6(55)fast_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output6(55)mid_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output6(55)mid_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output6(55)slow_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output6(55)slow_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output9_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output9_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output10_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output10_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output11_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-11-13\\looming\\output11_left.mp4"

# PathVal_r = r"E:\\CSDIY\\F_discussion\\LGMD_video\\grating.avi"
# PathVal_l = r"E:\\CSDIY\\F_discussion\\LGMD_video\\grating.avi"


# PathVal_r = r"E:\\CSDIY\\F_discussion\\LGMD_video\\实验视频\\实验视频\\synthetic stimuli\\tf5-sf5.avi"
# PathVal_l = r"E:\\CSDIY\\F_discussion\\LGMD_video\\实验视频\\实验视频\\synthetic stimuli\\tf5-sf5.avi"

# PathVal_r = r"E:\\CSDIY\\F_discussion\\LGMD_video\\实验视频\\实验视频\\synthetic stimuli\\tf50-sf50.avi"
# PathVal_l = r"E:\\CSDIY\\F_discussion\\LGMD_video\\实验视频\\实验视频\\synthetic stimuli\\tf50-sf50.avi"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\Passing_By_right-test6.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\Passing_By_left-test6.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\image1-right-test1.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\image1-left-test1.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\image1-left-translating-5.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\image1-left-translating-5.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\image1-left-translating-15.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\image1-left-translating-15.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\turing2_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\turing2_right.mp4"

# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\looming1_left.mp4"
# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\looming1_right.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\looming6_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\looming6_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\looming5_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\looming5_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\looming4_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\looming4_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\looming3_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\looming3_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\looming2_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\looming2_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\60-60_looming_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\60-60_looming_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\60-40_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Bi-Video-robot\\video_robot\\60-40_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\video_20241206\\新建文件夹 (24)\\output_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\video_20241206\\新建文件夹 (24)\\output_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\video_20241206\\新建文件夹 (25)\\output_right.mp4"
# PathVal_l = r"G:\\G-CSDIY\\video_20241206\\新建文件夹 (25)\\output_left.mp4"

# PathVal_r = r"G:\\G-CSDIY\\video_20241128\\新建文件夹 (8)\\output_right (8).mp4"
# PathVal_l = r"G:\\G-CSDIY\\video_20241128\\新建文件夹 (8)\\output_left (8).mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-right-translating-30.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-left-translating-30.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-right-translating-15.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-left-translating-15.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-right-translating-5.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-left-translating-5.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-right-loomging-30.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-left-loomging-30.mp4"

PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-right-loomging-15.mp4"
PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-left-loomging-15.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-right-loomging-5.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-left-loomging-5.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-right-loomging-1.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-left-loomging-1.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-right-loomging-10.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-left-loomging-10.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-right-loomging-20.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-left-loomging-20.mp4"


# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-right-translating-1.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-left-translating-1.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-right-translating-10.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-left-translating-10.mp4"

# PathVal_r = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-right-translating-20.mp4"
# PathVal_l = r"G:\\G-CSDIY\\Panoramic natural scene video\\1.20\\imageC-left-translating-20.mp4"


# PathVal = r"E:\\CSDIY\\F_discussion\\LGMD_video\\实验视频\\实验视频\\ball stimuli\\123.mp4"

# PathVal = r"E:\\CSDIY\\F_discussion\\LGMD_video\\实验视频\\实验视频\\ball stimuli\\black high app4.mp4"

# PathVal_r = r"E:\\CSDIY\\IJCNN\\left_camera_projection.mp4"

# PathVal_l = r"E:\\CSDIY\\IJCNN\\right_camera_projection.mp4"

# PathVal = r"E:\\CSDIY\\IJCNN\\center_camera_projection.mp4"

# PathVal_r = r"E:\\CSDIY\\F_discussion\\LGMD_video\\t-0-255.avi"
if __name__ == '__main__':
    main(PathVal_l,PathVal_r)

