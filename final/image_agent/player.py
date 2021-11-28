import numpy as np
class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None

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
        return ['tux'] * num_players

    def act(self, player_state, player_image, puck_location):
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
        actions = []
        for i in range(self.num_players):
          steer_gain=2
          skid_thresh = 0.5
          target_vel = 25
          current_vel = np.linalg.norm(player_state[i]['kart']['velocity'])
          action = {}

          # if current_vel <= 1.0 and self.t - self.last_rescue > 30:
          #   action['rescue'] = True
          #   self.last_rescue = self.t

          # normalize location
          proj = np.array(player_state[i]['camera']['projection']).T
          view = np.array(player_state[i]['camera']['view']).T
          aim_point_world = puck_location['location']
          p = proj @ view @ np.array(list(aim_point_world) + [1])
          aim_point = np.array([p[0] / p[-1], -p[1] / p[-1]])

          forward_vector = [player_state[i]['kart']['front'][k] - player_state[i]['kart']['location'][k] for k in range(3)]
          puck_vector = [aim_point_world[k] - player_state[i]['kart']['location'][k] for k in range(3)]
          angle = np.arctan2(forward_vector[-1]*puck_vector[0] - forward_vector[0]*puck_vector[-1], forward_vector[0]*puck_vector[0] + forward_vector[-1]*puck_vector[-1])

          if (abs(aim_point[0]) > 1 or abs(aim_point[1]) > 1 or abs(angle) > np.pi / 2):
            aim_point[0] = 0
            aim_point[1] = 1
          

          puck_unknown = aim_point[1] > 0.5

          def chase_ball():
            steer_angle = steer_gain * aim_point[0]
            # print(current_vel)
            # if current_vel <= 10.5:
            #   self.low_speeds[i] += 1
            # else:
            #   self.low_speeds[i] = 0

            # if aim_point[0] < -1 or aim_point[0] > 1 or aim_point[1] > 1 or aim_point[1] < -1:
            #   action['brake'] = True
            #   current_vel = float('inf')
            #   steer_angle = -1 if steer_angle > 0 else 1

            # if self.low_speeds[i] > 15 and self.low_speeds[i] < 120:
            #   action['brake'] = True
            #   current_vel = float('inf')
            #   steer_angle = -1 if steer_angle > 0 else 1

            # Compute accelerate
            # if current_vel <= 1.3:
            #   action['brake'] = True
            #   current_vel = float('inf')
            #   steer_angle = 1
            #   print(aim_point)

            action['acceleration'] = 1.0 if current_vel < target_vel else 0.0

            # Compute steering
            action['steer'] = np.clip(steer_angle * steer_gain, -1, 1)

            # Compute skidding
            if abs(steer_angle) > skid_thresh:
              action['drift'] = True
            else:
              action['drift'] = False

            action['nitro'] = True

          def back_up():
            steer_angle = 0
            action['acceleration'] = 0.0
            action['steer'] = steer_angle
            action['brake'] = True

          if (puck_unknown):
            back_up()
          else:
            chase_ball()
          actions.append(action)
          self.t += 1
        # print(self.low_speeds)
        return actions
