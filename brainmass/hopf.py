import brainscale
import brainstate
import jax.numpy as jnp
import jax

from .noise import OUProcess

__all__ = [
    'HopfModel',
]

class HopfModel(brainstate.nn.Dynamics):
    """
    The adaptive linear-nonlinear (aln) cascade model is a low-dimensional 
    population model of spiking neural networks. Mathematically, 
    it is a dynamical system of non-linear ODEs. 
    The dynamical variables of the system simulated in the aln model describe the average 
    firing rate and other macroscopic variables of a randomly connected, 
    delay-coupled network of excitatory and inhibitory adaptive exponential 
    integrate-and-fire neurons (AdEx) with non-linear synaptic currents.
    
    .. math::
    \frac{\mathrm{d}z}{\mathrm{d}t} = (a + \mathrm{i}\omega)\,z - \beta\,|z|^{2}z + I_{\text{ext}}(t)
    with :math:`|z|^{2} = x^{2} + y^{2}`.  
    Split into real/imaginary parts the system reads

    .. math::
    \begin{aligned}
    \dot x &= (a - \beta\,r)\,x - \omega\,y + coupled_x + I_{x}(t) \\
    \dot y &= (a - \beta\,r)\,y + \omega\,x + coupled_y + I_{y}(t)
    \end{aligned}
    \quad\text{with}\quad r = x^{2} + y^{2}.

    Parameters
    ----------
    x, y : dynamical variables
    Real and imaginary components of the oscillator (firing-rate analogue).
    a : bifurcation parameter
    > 0  →  limit-cycle (oscillatory);  ≤ 0  →  stable focus (silent).
    ω : angular frequency
    Intrinsic oscillation frequency (rad s⁻¹).
    β : nonlinear saturation coefficient
    Sets the limit-cycle amplitude (√ a/β ).
    K_gl : global coupling gain
    Scales diffusive input from other nodes.
    I_x, I_y : external inputs
    Additive currents (noise, coupling, stimulus) acting on x and y.
    coupled_x, coupled_y : coupling mechanism for neural network modules
 
    """

    def __init__ (
        self,
        in_size: brainstate.typing.Size,

        a : brainstate.typing.ArrayLike = 0.25,   # Hopf bifurcation parameter
        w : brainstate.typing.ArrayLike = 0.2,    # Oscillator frequency
        K_gl : brainstate.typing.ArrayLike = 1.0,    # global coupling strength
        β : brainstate.typing.ArrayLike = 1.0,    # nonlinear saturation coefficient

        # noise
        noise_x : OUProcess = None,
        noise_y : OUProcess = None,
            
    ):
        super().__init__(in_size=in_size)

        self.a = brainstate.init.param(a, self.varshape)
        self.w = brainstate.init.param(w, self.varshape)
        self.K_gl = brainstate.init.param(K_gl, self.varshape)
        self.β = brainstate.init.param(β, self.varshape)
        self.noise_x = noise_x
        self.noise_y = noise_y
        
    def init_state(self, batch_size=None, **kwargs):
        size = self.varshape if batch_size is None else (batch_size,) + self.varshape
        self.x = brainscale.ETraceState(jnp.zeros(size))
        self.y = brainscale.ETraceState(jnp.zeros(size))

    def reset_state(self, batch_size=None, **kwargs):
        size = self.varshape if batch_size is None else (batch_size,) + self.varshape 
        # initial values of the state variables
        self.x.value = brainstate.init.param(jnp.zeros, size)
        self.y.value = brainstate.init.param(jnp.zeros, size)

    def dynamics(self, x, y, coupled_x, ext_x, coupled_y, ext_y):
        r = x**2 + y**2
        dx_dt = (self.a - self.β * r) * x - self.w * y + coupled_x + ext_x
        dy_dt = (self.a - self.β * r) * y + self.w * x + coupled_y + ext_y
        return dx_dt, dy_dt
    
    def update(self, coupled_x, coupled_y, ext_x=None, ext_y=None):
        ext_x = 0. if ext_x is None else ext_x
        ext_y = 0. if ext_y is None else ext_y

        # add noise
        if self.noise_x is not None:
            assert isinstance(self.noise_y, OUProcess), "noise_y must be an OUProcess if noise_x is not None"
            ext_x += self.noise_x()

        if self.noise_y is not None:
            assert isinstance(self.noise_x, OUProcess), "noise_x must be an OUProcess if noise_y is not None"
            ext_y += self.noise_y()

        dx_dt, dy_dt = self.dynamics(self.x.value, self.y.value, coupled_x, ext_x, coupled_y, ext_y)
        dt = brainstate.environ.get_dt()
        x_next = self.x.value + dt * dx_dt
        y_next = self.y.value + dt * dy_dt
        # update the state
        # x, y = brainstate.nn.exp_euler_step(self.dynamics, self.x.value, self.y.value, coupled_x, ext_x, coupled_y, ext_y)
        self.x.value = x_next
        self.y.value = y_next
        return x_next, y_next
