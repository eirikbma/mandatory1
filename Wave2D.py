import numpy as np
import sympy as sp
from scipy import sparse

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation"""


    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""
        xi = np.linspace(0, 1, N + 1)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=sparse)
        return xij, yij

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        D2 = sparse.diags([1., -2., 1.], [-1, 0, 1], (N + 1, N + 1), format="lil")#Using a normal matrix here since we don't need so solve a linear algebra system.
        D2*=N**2
        return D2

    @property
    def w(self):
        """Return the dispersion coefficient"""
        return self.c*np.pi*np.sqrt(self.mx**2+self.my**2)

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int) -> tuple[np.ndarray, np.ndarray]:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        U0, U1: (N+1)**2 matricies for timestep 0 and 1
        """
        xij, yij = self.create_mesh(N)
        ue = self.ue(mx, my)
        ue_func = sp.lambdify((x, y, t), ue, "numpy")
        U0 = ue_func(xij, yij, 0)

        D2=self.D2(N)
        U1 = U0 + self.c**2*self.dt**2/2*(D2@U0+U0@D2.T)

        self.apply_bcs(U1)

        return U0, U1

    @property
    def dt(self) -> float:
        """Return the time step"""
        return self.cfl/(self.c*self.N)

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        xij,yij=self.create_mesh(self.N)
        h = 1/self.N

        ue = self.ue(self.mx, self.my)
        ue_func = sp.lambdify((x, y, t), ue, "numpy")
        ueij = ue_func(xij, yij, t0)

        return np.sqrt(h**2 * np.sum((ueij - u)**2))
        
    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """
        u[:,0]=0
        u[:,-1]=0
        u[0,:]=0
        u[-1,:]=0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.N = N #making some variables available to the entire class
        self.c = c
        self.cfl = cfl
        self.mx = mx
        self.my = my

        D2=self.D2(N)

        data={}

        U0,U1=self.initialize(N,mx,my)

        data[0]=U0

        

        if store_data==1:
            data[1]=U1
        elif store_data==-1:
            err=[]
            err.append(self.l2_error(U0,0))
            err.append(self.l2_error(U1,self.dt))


        for n in range(2,Nt+1):
            U2=2*U1-U0+(c*self.dt)**2*(D2@U1+U1@D2.T)
            self.apply_bcs(U2)
            if store_data==-1:
                err.append(self.l2_error(U2,n*self.dt))
            U0=U1 
            U1=U2
            if store_data>0 and n%store_data==0:
                data[n]=U2

        if store_data>0:
            return data
        elif store_data==-1:
            return 1/N, err


    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        D2 = sparse.diags([1., -2., 1.], [-1, 0, 1], (N + 1, N + 1), format="lil")#Using a normal matrix here since we don't need so solve a linear algebra system, so don't need the vec-trick.

        D2[0,1]=2#Neumann conditions
        D2[-1,-2]=2
        D2*=N**2
        return D2

    def ue(self, mx: int, my: int) -> sp.Expr:
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)

    def apply_bcs(self, u: np.ndarray):
        pass #Nothing needed to do on the boundary, all is handled by the difference in the D2-matrix


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    sol=Wave2D()
    _,err=sol(N=100,Nt=100,mx=2,my=2,cfl=1/np.sqrt(2),store_data=-1)
    assert max(err)<1e-12, max(err)

    sol=Wave2D_Neumann()
    _,err=sol(N=100,Nt=100,mx=2,my=2,cfl=1/np.sqrt(2),store_data=-1)
    assert max(err)<1e-12, max(err)
    

if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    print("convergence tests passed!")

    test_exact_wave2d()
    print("Test for exact solution passed!")


    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    frames = []

    sol = Wave2D_Neumann()
    xij,yij=sol.create_mesh(100)
    data=sol(N=100,Nt=100,cfl=1/np.sqrt(2),mx=2,my=2, store_data=5)

    for n, val in data.items():
        frame = ax.plot_wireframe(xij, yij, val, rstride=2, cstride=2)
        #frame = ax.plot_surface(xij, yij, val, vmin=-0.5*data[0].max(),
        #                        vmax=data[0].max(), cmap=cm.coolwarm,
        #                        linewidth=0, antialiased=False)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                    repeat_delay=1000)

   
    ani.save('neumannwave.gif', writer='pillow', fps=5)
    plt.show()
