"""Flappy-like continuous scrolling RL task with randomized weather and gaps.

Stylized 2D physics, not realistic butterfly aerodynamics.
Each level increases difficulty; reward is not a guarantee of infinite survival.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

LEVELS=("calm","wind","rain","obstacles","storm")
PARAMS={"calm":(0.0,0.0,0),"wind":(0.7,0.0,0),
        "rain":(0.9,0.25,0),"obstacles":(0.9,0.0,1),
        "storm":(1.5,0.4,1)}

class FlappyButterflyEnv(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self,level="calm",max_steps=1200):
        super().__init__()
        if level not in LEVELS: raise ValueError(level)
        self.level=level
        self.max_steps=max_steps
        self.action_space=spaces.Box(0,1,(1,),dtype=np.float32)
        # height, vertical velocity, current lift, wind, next gap position,
        # next gap center, gap half-height, distance to next obstacle
        self.observation_space=spaces.Box(-1,1,(8,),dtype=np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        r=self.np_random
        self.y=float(r.uniform(1.1,2.1))
        self.vy=float(r.uniform(-0.35,0.35))
        self.x=0.0
        self.lift=0.0
        self.wind=0.0
        self.steps=0
        self.passed=0
        self.gates=[]
        self.wind_scale,self.rain_drag,self.with_gates=PARAMS[self.level]
        self.wind_scale*=float(r.uniform(0.75,1.25))
        self.rain_drag*=float(r.uniform(0.75,1.25))
        if self.with_gates:
            for i in range(4): self._new_gate(3.0+i*2.8)
        return self._obs(),self._info(False)

    def _new_gate(self,x):
        r=self.np_random
        center=float(r.uniform(1.05,2.95))
        half=float(r.uniform(0.62,0.88) if self.level=="obstacles" else r.uniform(0.55,0.78))
        self.gates.append((float(x),center,half))

    def _next_gate(self):
        return next(((x,c,h) for x,c,h in self.gates if x+0.16>=self.x),(self.x+4,2,1.5))

    def _obs(self):
        gx,gc,gh=self._next_gate()
        return np.clip(np.array([(self.y-2)/2,self.vy/5,self.lift,
           self.wind/3,(gx-self.x)/4,(gc-self.y)/2,gh/2,
           (gx-self.x)/4],dtype=np.float32),-1,1)

    def _info(self,crashed):
        return {"crashed":bool(crashed),"passed":self.passed,"x":float(self.x),
            "y":float(self.y),"vy":float(self.vy),"wind":float(self.wind),
            "gates":list(self.gates),"level":self.level}

    def step(self,action):
        a=np.asarray(action,dtype=float)
        if a.shape!=(1,) or not np.isfinite(a).all(): raise ValueError("invalid action")
        a=float(np.clip(a[0],0,1))
        self.steps+=1
        dt=0.04
        # Delayed wing actuation and bounded vertical force.
        self.lift+=0.18*(a-self.lift)
        if self.wind_scale:
            self.wind=float(np.clip(0.96*self.wind+
                self.np_random.normal(0,0.13*self.wind_scale),-3,3))
        else:self.wind=0.0
        acceleration=17*self.lift-9.81-(0.45+self.rain_drag)*(self.vy-self.wind)
        self.vy=float(np.clip(self.vy+dt*acceleration,-7,7))
        previous_x=self.x
        self.y+=dt*self.vy
        self.x+=dt*1.7
        crashed=bool(self.y<=0.15 or self.y>=3.85)
        gates_passed_this_step=0
        for gx,gc,gh in self.gates:
            if abs(gx-self.x)<0.19 and abs(self.y-gc)>gh-0.12:
                crashed=True
            if previous_x<gx<=self.x and not crashed:
                self.passed+=1
                gates_passed_this_step+=1
        # Keep the course scrolling: recycle gates with fresh randomized gaps.
        self.gates=[g for g in self.gates if g[0]>=self.x-0.5]
        if self.with_gates and len(self.gates)<4:
            self._new_gate(max([g[0] for g in self.gates],default=self.x)+
                           float(self.np_random.uniform(2.4,3.3)))
        terminated=crashed
        truncated=self.steps>=self.max_steps and not terminated
        # Dense survival objective; bonus for clearing pipes, penalty for crashing.
        reward=0.05-0.005*abs(self.vy)-0.002*a*a
        # Local shaping: approach the center of the next gap only when nearby.
        # Moving forward is automatic, so this does not reward staying in place.
        if self.with_gates and not crashed:
            gx,gc,gh=self._next_gate()
            if 0 < gx-self.x < 1.5:
                reward-=0.015*abs(self.y-gc)
        if crashed:reward-=15
        reward+=3*gates_passed_this_step
        return self._obs(),float(reward),terminated,truncated,self._info(crashed)
