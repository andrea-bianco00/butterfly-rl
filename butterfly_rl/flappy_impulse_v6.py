"""V6: a single fixed-difficulty Flappy Bird task with impulse flaps.

NEW environment and physics. V5 policies MUST NOT be reused: actions now
mean discrete flap decisions, not continuous lift. Obstacles remain stochastic
across episodes but difficulty parameters do not change during training.
"""
import numpy as np
from gymnasium import spaces
from .curriculum_v5 import CurriculumFlappyEnvV5
from .curriculum_v4 import swept_collision,DT,HORIZONTAL_SPEED

class FlappyImpulseEnvV6(CurriculumFlappyEnvV5):
    def __init__(self,max_steps=1200):
        super().__init__(scenario="normal",max_steps=max_steps)
        self.action_space=spaces.Discrete(2)
        self.observation_space=spaces.Box(-1,1,(8,),dtype=np.float32)
    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        self.lift=0.0
        return self._obs(),info
    def _obs(self):
        gx,gc,gh=self._next_gate()
        return np.clip(np.array([(self.y-2)/2,self.vy/5,0.0,
           self.wind/3,(gx-self.x)/4,(gc-self.y)/2,gh/2,
           (gx-self.x)/4],dtype=np.float32),-1,1)
    def step(self,action):
        a=int(np.asarray(action).item())
        if a not in (0,1):raise ValueError("Flap must be 0 or 1")
        self.steps+=1
        if self.wind_scale:
            self.wind=float(np.clip(.96*self.wind+self.np_random.normal(0,.13*self.wind_scale),-3,3))
        else:self.wind=0.0
        if a==1:self.vy=3.0
        acceleration=-9.81-(.45+self.rain_drag)*(self.vy-self.wind)
        self.vy=float(np.clip(self.vy+DT*acceleration,-7,7))
        old_x,old_y=self.x,self.y
        self.y+=DT*self.vy
        self.x+=DT*HORIZONTAL_SPEED
        crashed=swept_collision(old_x,old_y,self.x,self.y,self.gates)
        credited=0
        if not crashed:
            for gx,gc,gh in self.gates:
                if gx not in self._credited_gates and self.x-.12>gx+.19:
                    self._credited_gates.add(gx);self.passed+=1;credited+=1
        self.gates=[g for g in self.gates if g[0]>=self.x-.5]
        if self.with_gates and len(self.gates)<4:
            self._new_gate(max([g[0] for g in self.gates],default=self.x)+float(self.np_random.uniform(2.4,3.3)))
        terminated=bool(crashed);truncated=bool(self.steps>=self.max_steps and not terminated)
        reward=.05-.004*abs(self.vy)-.004*a+3*credited-(15 if crashed else 0)
        return self._obs(),float(reward),terminated,truncated,self._info(crashed)
