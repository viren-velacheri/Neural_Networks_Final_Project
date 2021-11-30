import numpy as np
from image_agent.planner import load_model
import torchvision.transforms.functional as TF
import torch
class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = load_model().eval().to(self.device)

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        self.last_rescue = 0
        self.t = 0
        self.low_speeds = [0] * num_players
        self.puck_unknown = [0] * num_players
        self.steer_point = [[0, 0] for i in range(num_players)]
        self.rescue = [0] * num_players
        self.puck_height = 0.3693891763687134 #empirically discovered
        self.goal = None
        self.aim_points = [[0, 0] for i in range(num_players)]
        self.last_known_puck_location = np.array([0, 0, 0])

        return ['tux'] * num_players

    def act(self, player_state, player_image, puck_location):
        # constants; move these...
        steer_gain=2
        skid_thresh = 0.5
        target_vel = 25
        too_close_threshold = [0, 0.3] 
        unknown_threshold = 0.9
        red_goal = np.array([0.0000, -64.5000])
        blue_goal = np.array([0.0000,  64.5000])

        def set_goal_locations():
          if player_state[0]['kart']['location'][2] < 0:
            self.goal = blue_goal
          else:
            self.goal = red_goal

        def world_to_screen(camera, world_loc):
          #code adapted from prev hw
          proj = np.array(camera['projection']).T
          view = np.array(camera['view']).T
          big_P = proj @ view 
          p = big_P @ np.array(list(world_loc) + [1])
          aim_point = np.array([p[0] / p[-1], -p[1] / p[-1]])
          return aim_point

        # inverse operation of world_to_screen
        def screen_to_world(camera, screen_loc):
          screen_y = screen_loc[1] * -1 #undo multiplication by -1 in world_to_screen
          screen_x = screen_loc[0] 
          proj = np.array(camera['projection']).T
          view = np.array(camera['view']).T
          big_P = proj @ view # same as in world_to_screen
          big_P_inverse = np.linalg.inv(big_P) #use inverse matrix to inverse projection
          world_y = self.puck_height #lucky for us, this is constant; this is what allows us to do this inverse projection
          distance_from_camera = (big_P_inverse[1][0]* screen_x + big_P_inverse[1][1]* screen_y + big_P_inverse[1][3] - (big_P_inverse[3][0]*screen_x*world_y + big_P_inverse[3][1]*screen_y*world_y + big_P_inverse[3][3]*world_y)) / (big_P_inverse[3][2] * world_y - big_P_inverse[1][2])
          screen_coordinates = np.array([screen_x, screen_y, distance_from_camera, 1 ])
          w = big_P_inverse @ screen_coordinates
          world_coordinates = np.array([w[0] / w[-1], w[1] / w[-1], w[2] / w[-1]])
          return world_coordinates

        def find_puck(screen_points):
          best_location = None
          best_confidence = -999999
          for i in range(self.num_players):
            #where does this player think the puck is?
            if (screen_points[i][1] > unknown_threshold):
              #not on screen, we have no clue!
              pass
            else:
              #we think we know where the ball is
              camera = player_state[i]['camera']
              puck_location = screen_to_world(camera, screen_points[i])
              # could probably use a smarter "confidence" function
              confidence = -1 * (screen_points[i][0] ** 2 + screen_points[i][1] ** 2) 
              if (confidence > best_confidence):
                best_location = puck_location
          return best_location

        def set_aim_points(puck_location=None):
          for i in range(self.num_players):
            if (puck_location is not None):
              # cheated ground-truth puck screen location
              aim_point = world_to_screen(player_state[i]['camera'], puck_location)

              forward_vector = [player_state[i]['kart']['front'][k] - player_state[i]['kart']['location'][k] for k in range(3)]
              puck_vector = [puck_location[k] - player_state[i]['kart']['location'][k] for k in range(3)]
              angle = np.arctan2(forward_vector[-1]*puck_vector[0] - forward_vector[0]*puck_vector[-1], forward_vector[0]*puck_vector[0] + forward_vector[-1]*puck_vector[-1])

              if (abs(aim_point[0]) > 1 or abs(aim_point[1]) > 1 or abs(angle) > np.pi / 2):
                aim_point[0] = 0
                aim_point[1] = 1
              
              self.aim_points[i] = aim_point
            else:
              # Model predicted aim point below
              self.aim_points[i] = self.model(TF.to_tensor(player_image[i])[None].to(self.device)).squeeze(0).cpu().detach().numpy()


        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # TODO: Change me. I'm just cruising straight
        if self.t == 0:
          set_goal_locations()

        # cheated ground-truth puck world location
        true_puck_location = puck_location['location']


        # don't pass in puck_location to use model
        set_aim_points(puck_location = true_puck_location)

        projected_puck_location = find_puck(self.aim_points) # this is None if puck offscreen for all players
        if (projected_puck_location is not None):
          self.last_known_puck_location = projected_puck_location

        # print("Actual coordinates : " + str(true_puck_location))
        # print("Predicted coordinates : " + str(projected_puck_location))

        actions = []
        for i in range(self.num_players):  
          current_vel = np.linalg.norm(player_state[i]['kart']['velocity'])
          action = {}

          aim_point = self.aim_points[i]
          puck_coordinates = self.last_known_puck_location

          # potentially useful features
          kart_front = np.array(player_state[i]['kart']['front'])[[0, 2]]
          kart_center = np.array(player_state[i]['kart']['location'])[[0, 2]]
          kart_direction = (kart_front-kart_center) / np.linalg.norm(kart_front-kart_center)
          kart_angle = np.arctan2(kart_direction[1], kart_direction[0])

          puck_center = np.array(puck_coordinates[[0,2]])

          goal_to_puck_direction = (puck_center - self.goal) / np.linalg.norm((puck_center - self.goal))
          
          kart_to_puck = (puck_center - kart_center)
          kart_to_puck_distance = np.linalg.norm(kart_to_puck)
          kart_to_puck_direction = kart_to_puck / kart_to_puck_distance

          # # features of soccer 
          # puck_center = torch.tensor(soccer_state['ball']['location'], dtype=torch.float32)[[0, 2]]
          # kart_to_puck_direction = (puck_center - kart_center) / torch.norm(puck_center-kart_center)
          # kart_to_puck_angle = torch.atan2(kart_to_puck_direction[1], kart_to_puck_direction[0]) 
          # kart_to_puck_angle_difference = limit_period((kart_angle - kart_to_puck_angle)/np.pi)

          # # features of score-line 
          # goal_line_center = torch.tensor(soccer_state['goal_line'][team_id], dtype=torch.float32)[:, [0, 2]].mean(dim=0)
          # kart_to_goal_line = (goal_line_center-puck_center) / torch.norm(goal_line_center-puck_center)
          # kart_to_goal_line_angle = torch.atan2(kart_to_goal_line[1], kart_to_goal_line[0]) 
          # kart_to_goal_line_angle_difference = limit_period((kart_angle - kart_to_goal_line_angle)/np.pi)
          
          if (aim_point[1] > 0.5):
            self.puck_unknown[i] += 1
          else:
            self.puck_unknown[i] = 0
            self.steer_point[i] = aim_point

            
          if (abs(current_vel) < 1.0):
            self.rescue[i] += 1
          else:
            self.rescue[i] = 0

          def attack_ball():
            def activation(distance, angle):
              abs_angle = abs(angle)
              b = 0.5 
              w1 = 0.05
              w2 = 0.5
              return b + w1 * distance + w2 * abs_angle

            player_state[i]['kart']['state'] = 'attack_ball'
            target = puck_center + goal_to_puck_direction * activation(kart_to_puck_distance, np.arctan2(kart_to_puck_direction, goal_to_puck_direction))
            target_coords = np.array([target[0], self.puck_height, target[1]])

            screen_target = world_to_screen(player_state[i]['camera'], target_coords)
            
            chase_point(screen_target)

            player_state[i]['kart']['state'] = 'attack_ball'
            return

          def chase_ball():
            chase_point(self.steer_point[i])

            player_state[i]['kart']['state'] = 'chase_ball'
            return

          def chase_point(target_point):
            steer_angle = steer_gain * target_point[0]

            # Compute acceleration
            action['acceleration'] = 1.0 if current_vel < target_vel else 0.0

            # Compute steering
            action['steer'] = np.clip(steer_angle * steer_gain, -1, 1)

            # Compute skidding
            if abs(steer_angle) > skid_thresh:
              action['drift'] = True
            else:
              action['drift'] = False

            action['nitro'] = True

            player_state[i]['kart']['target'] = target_point
            player_state[i]['kart']['state'] = 'chase_point'
            return

          def back_up(steer_angle = 0):
            action['acceleration'] = 0.0
            action['steer'] = np.clip(steer_angle * steer_gain, -1, 1)
            action['brake'] = True

            player_state[i]['kart']['state'] = 'back_up'
            return

          def rescue():
            action['rescue'] = True

            player_state[i]['kart']['state'] = 'rescue'
            return

          def sleep():
            action['acceleration'] = 0.0
            if (current_vel >= 0):
              action['brake'] = True
            else:
              action['brake'] = False
            
            player_state[i]['kart']['state'] = 'sleep'
            return

          if (self.puck_unknown[i] > 10):
            back_up()
          else:
            if (self.steer_point[i][1]) < too_close_threshold[0] and abs(self.steer_point[i][0]) < 3*abs(self.steer_point[i][1]):
              attack_ball()
            else:
              back_up(-1 * self.steer_point[i][0])
          
          if (self.rescue[i] > 30):
            rescue()

          # temp
          if (i != 0):
            sleep()

          actions.append(action)
          self.t += 1
        # print(self.low_speeds)
        return actions
