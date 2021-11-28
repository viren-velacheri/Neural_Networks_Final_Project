from .utils import load_recording
import numpy as np
from os import path

data = load_recording('./my_data.pkl')

output_path = './data/'

frame_count = 0
for frame in data:
  team1_images = frame['team1_images']
  team1_state = frame['team1_state']
  soccer_state = frame['soccer_state']
  for i in range(len(team1_images)):
    img = team1_images[i]

    from PIL import Image, ImageDraw
    if (isinstance(img, np.ndarray)):
      image = Image.fromarray(img)
    else:
      image = img
    
    # # normalize location
    proj = np.array(team1_state[i]['camera']['projection']).T
    view = np.array(team1_state[i]['camera']['view']).T
    aim_point_world = soccer_state['ball']['location']
    p = proj @ view @ np.array(list(aim_point_world) + [1])
    aim_point = np.array([p[0] / p[-1], -p[1] / p[-1]])

    forward_vector = [team1_state[i]['kart']['front'][k] - team1_state[i]['kart']['location'][k] for k in range(3)]
    puck_vector = [aim_point_world[k] - team1_state[i]['kart']['location'][k] for k in range(3)]
    angle = np.arctan2(forward_vector[-1]*puck_vector[0] - forward_vector[0]*puck_vector[-1], forward_vector[0]*puck_vector[0] + forward_vector[-1]*puck_vector[-1])

    if (abs(aim_point[0]) > 1 or abs(aim_point[1]) > 1 or abs(angle) > np.pi / 2):
      aim_point[0] = 0
      aim_point[1] = 1

    label = aim_point

    fn = path.join(output_path, '%05d_%d' % (frame_count, i))
    image.save(fn + '.png')
    with open(fn + '.csv', 'w') as f:
      f.write('%0.1f,%0.1f' % tuple(label))

  team2_images = frame['team2_images']
  team2_state = frame['team2_state']
  soccer_state = frame['soccer_state']
  for i in range(len(team2_images)):
    img = team2_images[i]

    from PIL import Image, ImageDraw
    if (isinstance(img, np.ndarray)):
      image = Image.fromarray(img)
    else:
      image = img
    
    # # normalize location
    proj = np.array(team2_state[i]['camera']['projection']).T
    view = np.array(team2_state[i]['camera']['view']).T
    aim_point_world = soccer_state['ball']['location']
    p = proj @ view @ np.array(list(aim_point_world) + [1])
    aim_point = np.array([p[0] / p[-1], -p[1] / p[-1]])

    forward_vector = [team2_state[i]['kart']['front'][k] - team2_state[i]['kart']['location'][k] for k in range(3)]
    puck_vector = [aim_point_world[k] - team2_state[i]['kart']['location'][k] for k in range(3)]
    angle = np.arctan2(forward_vector[-1]*puck_vector[0] - forward_vector[0]*puck_vector[-1], forward_vector[0]*puck_vector[0] + forward_vector[-1]*puck_vector[-1])

    if (abs(aim_point[0]) > 1 or abs(aim_point[1]) > 1 or abs(angle) > np.pi / 2):
      aim_point[0] = 0
      aim_point[1] = 1

    label = aim_point

    fn = path.join(output_path, '%05d_%d' % (frame_count, i + len(team1_images)))
    image.save(fn + '.png')
    with open(fn + '.csv', 'w') as f:
      f.write('%0.1f,%0.1f' % tuple(label))

  
  frame_count += 1





print(data)