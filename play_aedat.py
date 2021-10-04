from dv import AedatFile
import cv2
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sdvs_N.functions_classes import *

def timestep(LR):
    with open(LR + '_timestep.txt', 'r') as f:
        lines = f.read()

    timestep = lines.split("\n")

    timestep = np.array([int(s) for s in timestep if s])
    return timestep

class evs_from_struct:
    def __init__(self, mat, LR):
        self.timestamp = timestep(LR)
        # self.timestamp = self.timestamp.reshape
        self.x = mat['x'][0:926816]
        self.y = mat['y'][0:926816]
        self.polarity = mat['polarity'][0:926816]


def main():

    with AedatFile('/Users/shanmugav/Desktop/NewFolder06:30/IBM-CVPR/DVS2_try/dvSave_left-2021_09_15_18_27_36.aedat4') as left, AedatFile('/Users/shanmugav/Desktop/NewFolder06:30/IBM-CVPR/DVS2_try/dvSave_right-2021_09_15_18_27_36.aedat4') as right :

        # list all the names of streams in the file
        print(left.names, right.names, left['events'].size, right['events'].size)

        l_events = np.hstack([packet for packet in left['events'].numpy()])
        r_events = np.hstack([packet for packet in right['events'].numpy()])
        # print(l_events[1]) # 9205397 # timestamp, x, y, polarity
        # print(l_events['timestamp'].size)
        # file1 = open('timestamp_l_events.txt', 'a')
        # for time in l_events['timestamp']:
        #     file1.write(str(time) + '\n')
        # file1.close()
        # print(l_events['polarity'])

        L = evs_from_struct(l_events, 'L')
        R = evs_from_struct(r_events, 'R')
        # print(len(L.timestamp), L.timestamp.shape, L.x.shape, len(L.x), L.y.shape, len(L.y), L.polarity.shape, len(L.polarity))

        print(L.x.shape, L.y.shape, L.polarity.shape, L.timestamp.shape)
        print(R.x.shape, R.y.shape, R.polarity.shape, R.timestamp.shape)
        print("minmax", min(L.x), max(L.x), min(L.y), max(L.y))

        POI = 1 # polarity of interest
        L_idxs_pol = [idx for idx in range(len(L.polarity)) if L.polarity[idx] == POI]
        R_idxs_pol = [idx for idx in range(len(R.polarity)) if R.polarity[idx] == POI]
        print(len(L_idxs_pol))

        L.x = np.array([L.x[i] for i in L_idxs_pol])
        L.y = np.array([L.y[i] for i in L_idxs_pol])
        L.polarity = np.array([L.polarity[i] for i in L_idxs_pol])
        L.timestamp = np.array([L.timestamp[i] for i in L_idxs_pol])

        R.x = np.array([R.x[i] for i in R_idxs_pol])
        R.y = np.array([R.y[i] for i in R_idxs_pol])
        R.polarity = np.array([R.polarity[i] for i in R_idxs_pol])
        R.timestamp = np.array([R.timestamp[i] for i in R_idxs_pol])

        print(L.x.shape, L.y.shape, L.polarity.shape, L.timestamp.shape)
        print(R.x.shape, R.y.shape, R.polarity.shape, R.timestamp.shape)
        print(len(L_idxs_pol), len(R_idxs_pol))
        print("minmax_new", min(L.x), max(L.x), min(L.y), max(L.y))

        num_ticks = min(max(L.timestamp), max(R.timestamp))
        print(num_ticks, max(L.timestamp))

        num_L_events = len(L.timestamp)
        num_R_events = len(R.timestamp)

        w = 60
        kernel_size = [2,2]
        dim = 320 + 2*w

        T_windows = [2, 10, 50]

        L_ev_idxs = np.arange(len(L.timestamp)) #[0, 1, 2, 3, 4, 5, 6, 7, ...926816]
                                                #[0, 0, 1, 1, 2, 2, 2, 2, ...
        #[0,0,0,1,1,1,2,2,2,3,4,5,6,6,7,7,7,8,8,8,8,9,9,9,9,9,10,10,10, 11,11,11,11, ....] # L.timestamp
        # [T,T,T,T,T,T,T,T,T,T,T,T,T<T,T,T,T<T,T,T,T,T<T<T,T<T<T<T<,T,T,F,F,F,F,F,F,F,F,F, ...] # L.timestamp <=i
        #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28, ...] # L_ev_idxs
        #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22] # L_past_evs

        # [-,-,-,-,-,-,-,-,-,-,-,-,-,-,13,14,15,16,17,18,19,20,21,22] # l_temp
        # time window = 9

        R_ev_idxs = np.arange(len(R.timestamp))

        denom = 10

        data = []
        depth_frames = []
        # iterate through each frame t
        for i in 1 + np.arange(num_ticks // denom):

            # allocate space for events to be drawn
            l_evs = np.zeros((dim, dim, np.shape(T_windows)[0]))
            r_evs = np.zeros((dim, dim, np.shape(T_windows)[0]))

            # identify which events have already occured
            L_past_events_idxs = L_ev_idxs[L.timestamp <= i]
            R_past_events_idxs = R_ev_idxs[R.timestamp <= i]
            # print("L_past_events_idxs, R_past_events_idxs", len(L_past_events_idxs), len(R_past_events_idxs))

            # store events in their allocated spaces according to the time window
            for t in range(np.shape(T_windows)[0]):
                # find which events have occured in time window t
                l_temp = L_past_events_idxs[L.timestamp[L_past_events_idxs] > max(0, i - T_windows[t])]
                r_temp = R_past_events_idxs[R.timestamp[R_past_events_idxs] > max(0, i - T_windows[t])]

                # print("l_temp, r_temp ", len(l_temp), len(r_temp))

                l_xs = L.x[l_temp].flatten() + w
                l_ys = L.y[l_temp].flatten() + w

                # print("l_xs, l_ys ", len(l_xs), len(l_ys))
                # print("array ", l_xs, l_ys)

                r_xs = R.x[r_temp].flatten() + w
                r_ys = R.y[r_temp].flatten() + w
                # print("r_xs, r_ys ", len(r_xs), len(r_ys))

                l_evs_temp = np.zeros((dim, dim))
                r_evs_temp = np.zeros((dim, dim))

                # print("l_evs_temp, r_evs_temp ", len(l_evs_temp), len(r_evs_temp) )

                l_evs_temp[l_xs, l_ys] = 1
                r_evs_temp[r_xs, r_ys] = 1

                l_evs[:, :, t] = l_evs_temp
                r_evs[:, :, t] = r_evs_temp

            # find all events that have taken place in the last timestep to get time surfaces
            l_temp = L_past_events_idxs[L.timestamp[L_past_events_idxs] > max(0, i - T_windows[t])]
            r_temp = R_past_events_idxs[R.timestamp[R_past_events_idxs] > max(0, i - T_windows[t])]

            l_xs = L.x[l_temp].flatten() + w
            l_ys = L.y[l_temp].flatten() + w

            r_xs = R.x[r_temp].flatten() + w
            r_ys = R.y[r_temp].flatten() + w

            # extract and contatentate time surfaces from different time windows
            TSs_l = get_neighborhood(l_xs, l_ys, w, l_evs)
            TSs_r = get_neighborhood(r_xs, r_ys, w, r_evs)

            # spatial scaling of the concatenated event patches from different time windows
            TSs_scaled_l = downscale(TSs_l, kernel_size)
            TSs_scaled_r = downscale(TSs_r, kernel_size)

            # pairwise comparison
            Dlr = dist_measure(TSs_scaled_l, TSs_scaled_r)

            # lr -> left events as rows / right events as columns
            # rl -> opposite
            lr_best_matches = [np.arange(np.shape(Dlr)[0])[Dlr[:, i] == np.max(Dlr[:, i])] for i in range(np.shape(Dlr)[1])]
            rl_best_matches = [np.arange(np.shape(Dlr)[1])[Dlr[i, :] == np.max(Dlr[i, :])] for i in range(np.shape(Dlr)[0])]

            winner_pairs = []  # left, right pairs
            for l_pix, r_pix_match in enumerate(lr_best_matches):

                for r_pix in r_pix_match:  # these are indices

                    # see if the match is reciprocated
                    if l_pix in rl_best_matches[r_pix]:
                        winner_pairs.append([l_pix, r_pix])

            temp_map = np.zeros((dim, dim))
            depth_map, X, Y, Z = disparity(winner_pairs, l_xs, l_ys, r_xs, r_ys, temp_map)
            data.append([X, Y, Z])
            depth_frames.append(depth_map)


                # every 5 ticks, do a sanity check
            if i%1 == 0:
                print(i, 'out of', num_ticks/denom)


        def animate(frame):
            f = depth_frames[frame]
            line.set_data(f)

        fig = plt.figure(10)
        line = plt.imshow([[]], extent=(0,10,0,10), cmap='viridis', clim=(0,1))
        anim = animation.FuncAnimation(fig, animate, frames=num_ticks//denom, interval=60)
        HTML(anim.to_jshtml())

        writervideo = animation.FFMpegWriter(fps=30)
        anim.save('depth_frames.mp4', writer=writervideo)

if __name__ == "__main__":
    main()