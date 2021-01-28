import pyglet

episode_num = 999
animation_length = 100
state_file = '/fs/clip-realspeech/projects/vid_game/data/old/experimental_oneshotgame5/exp/state_out_run15.txt'
reward_file = '/fs/clip-realspeech/projects/vid_game/data/old/experimental_oneshotgame5/exp/reward_out_run15.txt'
action_file = '/fs/clip-realspeech/projects/vid_game/data/old/experimental_oneshotgame5/exp/action_out_run15.txt'
location_file = '/fs/clip-realspeech/projects/vid_game/data/old/experimental_oneshotgame5/exp/location_out_run15.txt'


class Animation():

    def __init__(self, episode_num, animation_length):
        self.actions = []
        self.rewards = []
        self.states = []
        self.locations = []
        self.index = 1

        self.rotation = 0
        self.current_reward = 0
        self.is_shoot=False
        self.episode_to_load = episode_num

        self.animation_length = animation_length
        self.compass = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

    def load_data(self, state_file, reward_file, action_file, locations_file):
        with open(state_file, 'r') as f:

            def what_state(s):
                if s[0]=='a':
                    return 0
                if s[0]=='b':
                    return 1
                if s[0]=='c':
                    return 2
                if s[0]=='d':
                    return 3

            input_data = f.read().splitlines()
            states = input_data[self.episode_to_load].split(' ')[1:]
            print(states)
            self.states = [what_state(s) for s in states]
            print(self.states)

        with open(action_file, 'r') as f:
            input_data = f.read().splitlines()
            self.actions = input_data[self.episode_to_load].split(' ')[1:]

        with open(reward_file, 'r') as f:
            input_data = f.read().splitlines()
            self.rewards = input_data[self.episode_to_load].split(' ')[1:]

        with open(locations_file, 'r') as f:
            input_data = f.read().splitlines()
            self.locations=input_data[self.episode_to_load].split(' ')[1:]
            self.starting_location = input_data[self.episode_to_load].split(' ')[0]

    def initiate_environment(self):
        self.index =-1
        self.update()

    def update(self):
        self.index += 1
        self.is_shoot = False
        if int(float(self.actions[self.index])) == 2:
            self.is_shoot=True

        self.current_reward += int(float(self.rewards[self.index]))

        self.rotation = self.compass.index(self.locations[self.index]) * 45

        print('-----------')
        print(self.locations[self.index])
        print(self.rotation)
        print(self.actions[self.index])
        print(self.is_shoot)



    def get_state(self):
        return self.states[self.index]

    def get_rotation(self):
        self.rotation = self.rotation%360
        return self.rotation

    def get_reward(self):
        return self.current_reward

    def get_shoot(self):
        return self.is_shoot

    def get_location(self):
        return self.compass.index(self.locations[self.index])

    def alien_dead(self):
        self.is_shoot=False




print('creating animation')
animation = Animation(episode_num,50)

print('loading data')
animation.load_data(state_file, reward_file, action_file, location_file)

print('initiating animation')
animation.initiate_environment()

print('opening window')
w = pyglet.window.Window()

print('loading images')
rocket_list = []
degrees = [0,45, 90, 135, 180, 225, 270, 315]
for x in degrees:
    rocket = pyglet.image.load('/fs/clip-realspeech/projects/vid_game/animation/rocket'+str(x)+'.jpeg')
    rocket.anchor_x = rocket.width // 2
    rocket.anchor_y = rocket.height // 2
    #print(rocket.get_texture().width)
    #print(rocket.get_texture().height)
    #rocket.width = 40#int(.2 * rocket.width)
    #rocket.height = int(0.2 * rocket.height)
    rocket_list.append(rocket)


aliens_locations = [(w.width//2, w.height-40), (w.width//2, 40), (w.width -40 , w.height//2), (w.width -40 , w.height//2)]
alien = pyglet.image.load('/fs/clip-realspeech/projects/vid_game/animation/alien.png')
alien_dead=False
new_alien=False
steps_since_dead=5
current_rocket=None
def generate_shoot_locations(width, height):
    shoot_locations = [(width*0.5, height*0.75),
                       (width*0.75, height*0.75),
                       (width*0.75, height*0.5),
                       (width*0.75, height*0.25),
                       (width*0.5, height*0.25),
                       (width*0.25, height*0.25),
                       (width*0.25, height*0.5),
                       (width*0.25, height*0.75)]
    return [(int(s[0]), int(s[1])) for s in shoot_locations]

shoot_locations = generate_shoot_locations(w.width, w.height)

@w.event
def on_key_press(symbol, modifiers):
    w.clear()
    global animation
    global alien_dead
    global steps_since_dead
    global new_alien
    steps_since_dead += 1
    if animation.get_shoot():
        alien_dead = True
        animation.alien_dead()
        new_alien=True
        steps_since_dead=0
    elif steps_since_dead==1:
        animation.update()
        alien_dead=False
    elif steps_since_dead==2:
        new_alien=False
    else:
        animation.update()



@w.event
def on_resize(width, height):
    global aliens_locations
    global shoot_locations
    aliens_locations = [(width // 2, height - 100), (width - 100, height // 2), (width // 2, 40),
                        (100, height // 2)]
    shoot_locations = generate_shoot_locations(width, height)

@w.event
def on_draw():
    w.clear()

    global animation
    global rocket_list
    global current_rocket
    label = pyglet.text.Label('Score: '+str(animation.get_reward()),
                              font_name='Times New Roman',
                              font_size=18,
                              x=75, y=w.height-15,
                              anchor_x='center', anchor_y='center')
    label.draw()
    if animation.get_shoot():
        x,y = shoot_locations[animation.get_location()]
        shoot = pyglet.text.Label('Shoot!',
                                  font_name='Times New Roman',
                                  font_size=18,
                                  x=x, y=y,
                                  anchor_x='center', anchor_y='center')
        shoot.draw()
    #rocket.rotation = animation.get_rotation()
    if not new_alien:
        current_rocket = rocket_list[degrees.index(animation.get_rotation())]
    alien_x, alien_y = aliens_locations[animation.get_state()]
    #print(alien_x, alien_y)
    if not alien_dead:
        alien.blit(x=alien_x, y = alien_y)
    #rocket = rocket.get_texture().get_transform(animation.get_rotation())
   # rocket = rocket.get_transform(rotate = animation.get_rotation())
    current_rocket.blit(x=w.width//2, y=w.height//2)
    #print(str(animation.get_rotation()))


pyglet.app.run()
