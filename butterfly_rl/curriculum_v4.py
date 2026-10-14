"""V4 environment: swept vertical collision checks, v3 kept unchanged.

The integration scheme and wind process are intentionally identical to v3.
Within a step the center follows a linear segment from old to new position.
This fixes end-point-only tunneling under that interpolation assumption.
It is not a continuous-time aerodynamics simulator.
"""
import numpy as np
from .curriculum_v3 import CurriculumFlappyEnv

RADIUS=0.12
WALL_HALF_WIDTH=0.19
LOWER=0.15
UPPER=3.85
DT=0.04
HORIZONTAL_SPEED=1.7

def swept_collision(x0,y0,x1,y1,gates):
    """Check the entire line segment of a circular player's center.

    Pipe opening is shrunk by player radius; horizontal slab is enlarged
    by player radius. Conservative axis-aligned Minkowski expansion.
    """
    if min(y0,y1)<=LOWER or max(y0,y1)>=UPPER:
        return True
    dx=x1-x0
    for gx,gc,gh in gates:
        left=gx-WALL_HALF_WIDTH-RADIUS
        right=gx+WALL_HALF_WIDTH+RADIUS
        if dx==0:
            if not left<=x0<=right:continue
            lo=hi=0.0
            hi=1.0
        else:
            lo=max(0.0,min((left-x0)/dx,(right-x0)/dx))
            hi=min(1.0,max((left-x0)/dx,(right-x0)/dx))
            if lo>hi:continue
        ya=y0+(y1-y0)*lo
        yb=y0+(y1-y0)*hi
        if min(ya,yb)<gc-gh+RADIUS or max(ya,yb)>gc+gh-RADIUS:
            return True
    return False

class CurriculumFlappyEnvV4(CurriculumFlappyEnv):
    """Same scenarios and observations as v3; new collision semantics."""

    def step(self,action):
        a=np.asarray(action,dtype=float)
        if a.shape!=(1,) or not np.isfinite(a).all():
            raise ValueError("invalid action")
        a=float(np.clip(a[0],0,1))
        self.steps+=1
        self.lift+=0.18*(a-self.lift)
        if self.wind_scale:
            self.wind=float(np.clip(0.96*self.wind+
                self.np_random.normal(0,0.13*self.wind_scale),-3,3))
        else:
            self.wind=0.0
        acceleration=17*self.lift-9.81-(0.45+self.rain_drag)*(self.vy-self.wind)
        self.vy=float(np.clip(self.vy+DT*acceleration,-7,7))
        old_x,old_y=self.x,self.y
        self.y+=DT*self.vy
        self.x+=DT*HORIZONTAL_SPEED
        crashed=swept_collision(old_x,old_y,self.x,self.y,self.gates)
        passed_this_step=0
        if not crashed:
            for gx,gc,gh in self.gates:
                if (gx not in self._credited_gates and
                    self.x-RADIUS>gx+WALL_HALF_WIDTH):
                    self._credited_gates.add(gx)
                    self.passed+=1
                    passed_this_step+=1
        self.gates=[g for g in self.gates if g[0]>=self.x-0.5]
        if self.with_gates and len(self.gates)<4:
            self._new_gate(max([g[0] for g in self.gates],default=self.x)+
                           float(self.np_random.uniform(2.4,3.3)))
        terminated=bool(crashed)
        truncated=bool(self.steps>=self.max_steps and not terminated)
        reward=0.05-0.005*abs(self.vy)-0.002*a*a
        if self.with_gates and not crashed:
            gx,gc,gh=self._next_gate()
            if 0<gx-self.x<1.5:
                reward-=0.015*abs(self.y-gc)
        if crashed:reward-=15
        reward+=3*passed_this_step
        return self._obs(),float(reward),terminated,truncated,self._info(crashed)
