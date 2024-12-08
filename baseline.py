# import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
from trajectory_map import TrajectoryMap
import pygame
import cv2

import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm
from natsort import natsorted
from sklearn.cluster import MiniBatchKMeans
import math
import faiss

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        super(KeyboardPlayerPyGame, self).__init__()
        
        # Variables for reading exploration data
        self.save_dir = "data/images1/"
        if not os.path.exists(self.save_dir):
            print(f"Directory {self.save_dir} does not exist, please download exploration data.")

        # Initialize SIFT detector
        # SIFT stands for Scale-Invariant Feature Transform
        self.sift = cv2.SIFT_create(nfeatures=100)
        self.bf = cv2.BFMatcher()
        # Load pre-trained sift features and codebook
        self.sift_descriptors, self.codebook = None, None
        self.database = None
        if os.path.exists("sift_descriptors.npy"):
            print("Sift descriptors is located at ", os.path.abspath("sift_descriptors.npy"))
            self.sift_descriptors = np.load("sift_descriptors.npy").astype(np.float32)
        if os.path.exists("codebook.pkl"):
            self.codebook = pickle.load(open("codebook.pkl", "rb"))
        if os.path.exists("database.pkl"):
            self.database = pickle.load(open("database.pkl", "rb"))
        if os.path.exists("database.npy"):
            self.database = np.load("database.npy").astype(np.float32)
        # Initialize database for storing VLAD descriptors of FPV
        self.goal = None
        self.targets = None

        self.trajectory_map = TrajectoryMap()
        self.faiss_index = None

    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            #  Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                else:
                    # If a key is pressed that is not mapped to an action, then display target images
                    self.show_target_images()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def display_img_from_id(self, id, window_name):
        """
        Display image from database based on its ID using OpenCV
        """
        path = self.save_dir + "image_" + str(id) + ".png"
        print(f"Displaying image from path: {path}")
        if os.path.exists(path):
            img = cv2.imread(path)
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
        else:
            print(f"Image with ID {id} does not exist")

    def compute_sift_features(self):
        """
        Compute SIFT features for images in the data directory
        """
        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.png')])
        sift_descriptors = list()
        for i, img in enumerate(tqdm(files, desc="Processing images")):
            # if i < 1480: continue
            img = cv2.imread(os.path.join(self.save_dir, img))
            # Pass the image to sift detector and get keypoints + descriptions
            # We only need the descriptors
            # These descriptors represent local features extracted from the image.
            _, des = self.sift.detectAndCompute(img, None)
            if des is None or len(des) == 0: 
                print(f"Image {img} at {i} has no descriptors")

                # raise ValueError(f"Image {img} at {i} has no descriptors")
                continue
            # Extend the sift_descriptors list with descriptors of the current image
            sift_descriptors.extend(des.astype(np.float32))
        return np.asarray(sift_descriptors)
    
    def get_VLAD(self, img):
        """
        Compute VLAD (Vector of Locally Aggregated Descriptors) descriptor for a given image
        """
        # We use a SIFT in combination with VLAD as a feature extractor as it offers several benefits
        # 1. SIFT features are invariant to scale and rotation changes in the image
        # 2. SIFT features are designed to capture local patterns which makes them more robust against noise
        # 3. VLAD aggregates local SIFT descriptors into a single compact representation for each image
        # 4. VLAD descriptors typically require less memory storage compared to storing the original set of SIFT
        # descriptors for each image. It is more practical for storing and retrieving large image databases efficicently.

        # Pass the image to sift detector and get keypoints + descriptions
        # Again we only need the descriptors
        _, des = self.sift.detectAndCompute(img, None)
        # We then predict the cluster labels using the pre-trained codebook
        # Each descriptor is assigned to a cluster, and the predicted cluster label is returned
        if des is None or len(des) == 0:
            return np.zeros(self.codebook.n_clusters * 128, dtype=np.float32)
        pred_labels = self.codebook.predict(des)
        # Get number of clusters that each descriptor belongs to
        centroids = self.codebook.cluster_centers_.astype(np.float32)
        # Get the number of clusters from the codebook
        k = self.codebook.n_clusters
        VLAD_feature = np.zeros([k, des.shape[1]], dtype=np.float32)

        # Loop over the clusters
        for i in range(k):
            # If the current cluster label matches the predicted one
            if np.sum(pred_labels == i) > 0:
                # Then, sum the residual vectors (difference between descriptors and cluster centroids)
                # for all the descriptors assigned to that clusters
                # axis=0 indicates summing along the rows (each row represents a descriptor)
                # This way we compute the VLAD vector for the current cluster i
                # This operation captures not only the presence of features but also their spatial distribution within the image
                VLAD_feature[i] = np.sum(des[pred_labels==i, :] - centroids[i], axis=0)
        VLAD_feature = VLAD_feature.flatten()
        # Apply power normalization to the VLAD feature vector
        # It takes the element-wise square root of the absolute values of the VLAD feature vector and then multiplies 
        # it by the element-wise sign of the VLAD feature vector
        # This makes the resulting descriptor robust to noice and variations in illumination which helps improve the 
        # robustness of VPR systems
        VLAD_feature = np.sign(VLAD_feature)*np.sqrt(np.abs(VLAD_feature))
        # Finally, the VLAD feature vector is normalized by dividing it by its L2 norm, ensuring that it has unit length
        VLAD_feature = VLAD_feature/np.linalg.norm(VLAD_feature)

        return VLAD_feature

    def get_neighbor(self, img):
        """
        Find the nearest neighbor in the database based on VLAD descriptor
        """
        # Get the VLAD feature of the image
        q_VLAD = self.get_VLAD(img).reshape(1, -1)
        # This function returns the index of the closest match of the provided VLAD feature from the database the tree was created
        # The '1' indicates the we want 1 nearest neighbor
        _, index = self.tree.query(q_VLAD, 1)
        return index[0][0]

    def pre_nav_compute(self):
        """
        Build BallTree for nearest neighbor search and find the goal ID
        """
        # Compute sift features for images in the database
        if self.sift_descriptors is None:
            print("Computing SIFT features...")
            self.sift_descriptors = self.compute_sift_features()
            np.save("sift_descriptors.npy", np.array(self.sift_descriptors, dtype=np.float32))
        else:
            print("Loaded SIFT features from sift_descriptors.npy")

        # KMeans clustering algorithm is used to create a visual vocabulary, also known as a codebook,
        # from the computed SIFT descriptors.
        # n_clusters = 64: Specifies the number of clusters (visual words) to be created in the codebook. In this case, 64 clusters are being used.
        # init='k-means++': This specifies the method for initializing centroids. 'k-means++' is a smart initialization technique that selects initial 
        # cluster centers in a way that speeds up convergence.
        # n_init=10: Specifies the number of times the KMeans algorithm will be run with different initial centroid seeds. The final result will be 
        # the best output of n_init consecutive runs in terms of inertia (sum of squared distances).
        # The fit() method of KMeans is then called with sift_descriptors as input data. 
        # This fits the KMeans model to the SIFT descriptors, clustering them into n_clusters clusters based on their feature vectors

        # TODO: try tuning the function parameters for better performance
        if self.codebook is None:
            print("Computing codebook...")
            # self.codebook = KMeans(n_clusters=128, init='k-means++', n_init=5, verbose=1).fit(self.sift_descriptors)
            self.codebook = MiniBatchKMeans(
                                n_clusters=64, batch_size=500, max_iter=200, n_init=5, verbose=1
                            ).fit(self.sift_descriptors.astype(np.float32))
            pickle.dump(self.codebook, open("codebook.pkl", "wb"))
        else:
            print("Loaded codebook from codebook.pkl")
        
        # get VLAD emvedding for each image in the exploration phase
        if self.database is None:
            self.database = []
            print("Computing VLAD embeddings...")
            exploration_observation = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.png')])
            for img in tqdm(exploration_observation, desc="Processing images"):
                img = cv2.imread(os.path.join(self.save_dir, img))
                VLAD = self.get_VLAD(img)
                self.database.append(VLAD)

            # pickle.dump(self.database, open("database.pkl", "wb"))
            np.save("database.npy", np.array(self.database, dtype=np.float32))
            
        # Build a BallTree for fast nearest neighbor search
        # We create this tree to efficiently perform nearest neighbor searches later on which will help us navigate and reach the target location
        # TODO: try tuning the leaf size for better performance
        # print("Building BallTree...")
        # print("Length of database: ", len(self.database)) 
        # tree = BallTree(self.database, leaf_size=64)
        
        # self.tree = tree
        self.database = np.load("database.npy").astype(np.float32)
        self.faiss_index = faiss.IndexFlatL2(self.database.shape[1])
        faiss.normalize_L2(self.database)
        self.faiss_index.add(self.database)
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors.")


    def pre_navigation(self):
        """
        Computations to perform before entering navigation and after exiting exploration
        """
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.pre_nav_compute()
    
    def display_images_from_indices(self, indices, offset=3, window_name="Dynamic Image Grid"):
        """
        Display images in a dynamic grid based on the number of indices.
        The grid is arranged with the best-fit row and column sizes.
        """
        images = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (255, 255, 255)  # White
        text_bg_color = (0, 0, 0)  # Black

        # Load and annotate images
        for idx in indices:
            # Choose the image based on the offset
            idx += offset
            path = self.save_dir + "image_" + str(idx) + ".png"
            if os.path.exists(path):
                img = cv2.imread(path)

                # Resize all images to a uniform size, e.g., 256x256
                img = cv2.resize(img, (256, 256))

                # Add ID text to the image
                text = f"ID: {idx}"
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x = 10
                text_y = 20

                # Draw background for text
                cv2.rectangle(img, (text_x, text_y - 15), (text_x + text_size[0] + 5, text_y + 5), text_bg_color, -1)

                # Add text overlay
                cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                images.append(img)
            else:
                print(f"Image with ID {idx} does not exist")

        # Compute grid size
        num_images = len(images)
        cols = math.ceil(math.sqrt(num_images))  # Number of columns
        rows = math.ceil(num_images / cols)     # Number of rows

        # Fill the grid with blank images if necessary
        blank_image = np.zeros((256, 256, 3), dtype=np.uint8)
        while len(images) < rows * cols:
            images.append(blank_image)

        # Create the grid
        grid_rows = []
        for r in range(rows):
            start_idx = r * cols
            end_idx = start_idx + cols
            row = cv2.hconcat(images[start_idx:end_idx])
            grid_rows.append(row)

        grid = cv2.vconcat(grid_rows)

        # Display the final grid
        cv2.imshow(window_name, grid)
        cv2.waitKey(1)

    def display_topk_matched_images(self, img, offset=3, k=6):
        q_VLAD = self.get_VLAD(img).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q_VLAD)
        # _, indices = self.tree.query(q_VLAD, k)
        _, indices = self.faiss_index.search(q_VLAD, k)
        self.display_images_from_indices(indices[0], offset=offset)
        
    def verified_score(self, img1_des, img2_des):
        """
        Compute the verified score between two images
        """
        if img1_des is None:
            print("The img1 is None")
            return 0
        if img2_des is None:
            print("One of the descriptors is None.")
            return 0
        img1_des = img1_des.astype(np.float32)
        img2_des = img2_des.astype(np.float32)
        matches = self.bf.knnMatch(img1_des, img2_des, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
        return len(good_matches)

    def get_best_neighbor(self, img):
        """
        Find the best neighbor in the database based on VLAD descriptor
        """
        # Get the VLAD feature of the image
        q_VLAD = self.get_VLAD(img).reshape(1, -1).astype(np.float32)
        _, des = self.sift.detectAndCompute(img, None)
        # This function returns the index of the closest match of the provided VLAD feature from the database the tree was created
        # The '1' indicates the we want 1 nearest neighbor
        # _, indices = self.tree.query(q_VLAD, 6)
        _, indices = self.faiss_index.search(q_VLAD, 6)
        max_score = 0
        best_neighbor = None
        for idx in indices[0]:
            image_path = os.path.join(self.save_dir, f"image_{str(idx)}.png")
            print(f"Attempting to load image from: {image_path}")  # Debugging line
            
            # Check if the image exists
            if not os.path.exists(image_path):
                print(f"Image with ID {idx} does not exist at path: {image_path}")
                

            _, curr_des = self.sift.detectAndCompute(cv2.imread(image_path), None)
            curr_score = self.verified_score(des, curr_des)
            if curr_score > max_score:
                best_neighbor = idx
                max_score = curr_score
        return best_neighbor

    def display_next_best_view(self, offset=3):
        """
        Display the next best view based on the current first-person view
        """

        # TODO: could you write this function in a smarter way to not simply display the image that closely 
        # matches the current FPV but the image tha can efficiently help you reach the target?

        # Get the neighbor of current FPV
        # In other words, get the image from the database that closely matches current FPV
        # index = self.get_neighbor(self.fpv)
        index = self.get_best_neighbor(self.fpv)
        # Display the image 5 frames ahead of the neighbor, so that next best view is not exactly same as current FPV
        self.display_img_from_id(index+offset, f'Next Best View')
        # Display the next best view id along with the goal id to understand how close/far we are from the goal
        print(f'Next View ID: {index+offset} || Goal ID: {self.goal}')

        # Show the next best view on the trajectory map
        # self.trajectory_map.show_dot(index+3)s

    def see(self, fpv):
        """
        Set the first-person view input
        """

        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        # This allows subsequent rendering of the first-person view image onto the pygame screen
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:
                # TODO: could you employ any technique to strategically perform exploration instead of random exploration
                # to improve performance (reach target location faster)?
                
                # Nothing to do here since exploration data has been provided
                pass
            
            # If in navigation stage
            elif self._state[1] == Phase.NAVIGATION:
                # TODO: could you do something else, something smarter than simply getting the image closest to the current FPV?
                
                if self.goal is None:
                    # Get the neighbor nearest to the front view of the target image and set it as goal
                    self.targets = self.get_target_images()
                    index = self.get_best_neighbor(self.targets[0])
                    # index = self.get_neighbor(self.targets[0])
                    self.goal = index
                    print(f'Goal ID: {self.goal}')

                    # Show the target image on the trajectory map
                    # self.trajectory_map.show_target(self.goal)
                                
                # Key the state of the keys
                keys = pygame.key.get_pressed()
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q]:
                    self.display_next_best_view()
                
                if keys[pygame.K_f]:
                    self.display_topk_matched_images(self.fpv)
                
                if keys[pygame.K_t]:
                    self.display_img_from_id(self.get_best_neighbor(self.targets[0])-3, "Target Image")
                    self.display_topk_matched_images(self.targets[0], offset=-3)

        # Display the first-person view image on the pygame screen
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
