import numpy as np
from typing import List
from vanta_aero_panel.elementary_flows import UniformFlow
from scipy.linalg import solve


class Panel:
    """
    Represents a single straight-line source panel defined by two endpoints.
    """

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        """
        Sets the endpoints of the panel and derives its geometric properties.

        Parameters
        ----------
        x1: float
            x-coordinate of the panel start point.
        y1: float
            y-coordinate of the panel start point.
        x2: float
            x-coordinate of the panel end point.
        y2: float
            y-coordinate of the panel end point.
        """
        # Panel endpoints and centre point
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.xc, self.yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # Panel length
        self.l = ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) ** 0.5

        # Panel orientation angles
        self.theta = np.atan2(y2 - y1, x2 - x1)  # Inclination angle
        self.delta = self.theta + np.pi / 2.0  # Normal angle
        self.beta = None  # Angle relative to freestream (set by system)

        # Outward unit normal components
        self.xn, self.yn = np.cos(self.delta), np.sin(self.delta)

        # Solution quantities (populated after solve)
        self.strength = None
        self.vn = None
        self.vt = None
        self.cp = None


class SourcePanelSystem:
    """
    Assembles and solves a source panel method for a closed body in a uniform flow.
    """

    def __init__(self, panels: List[Panel], uniform_flow: UniformFlow):
        """
        Sets up the panel system and computes each panel's angle relative to the freestream.

        Parameters
        ----------
        panels: List[Panel]
            Ordered list of Panel objects discretising the body surface.
        uniform_flow: UniformFlow
            Freestream flow conditions.
        """
        # Panels and freestream
        self.panels = panels
        self.n_panels = len(panels)
        self.uniform_flow = uniform_flow

        # Angle of each panel normal relative to the freestream direction
        self.beta = [panel.delta - self.uniform_flow.alpha for panel in self.panels]

        # Solution quantities (populated after solve)
        self.strengths = None
        self.vn = None
        self.vt = None
        self.cp = None

    @staticmethod
    def integration_term(
        a1: float, a2: float, a3: float, a4: float, a5: float, s: float
    ) -> float:
        """
        Evaluates the analytic integral of a source panel's influence over its length.

        Parameters
        ----------
        a1: float
            Geometric coefficient derived from panel and field-point coordinates.
        a2: float
            Geometric coefficient equal to the squared distance from the field point
            to the panel start.
        a3: float
            Trigonometric coefficient for the logarithmic term.
        a4: float
            Trigonometric coefficient for the arctangent term.
        a5: float
            Geometric coefficient equal to the perpendicular distance from the field
            point to the panel line.
        s: float
            Panel length.

        Returns
        -------
        float
            Value of the integrated influence coefficient.
        """
        log_term = 0.5 * a3 * np.log((s**2 + 2 * a1 * s + a2) / a2)

        if abs(a5) < 1e-10:
            atan_term = 0.0
        else:
            atan_term = ((a4 - a1 * a3) / a5) * (
                np.atan2(s + a1, a5) - np.atan2(a1, a5)
            )

        return (1 / (2 * np.pi)) * (log_term + atan_term)

    def solve(self):
        """
        Solves for panel source strengths and computes surface velocities and
        pressure coefficients.
        """
        # Preallocate influence coefficient and RHS matrices
        an, at = (
            np.zeros((self.n_panels, self.n_panels)),
            np.zeros((self.n_panels, self.n_panels)),
        )
        bn, bt = np.zeros(self.n_panels), np.zeros(self.n_panels)

        # Populate influence coefficient and RHS matrices
        for i in range(self.n_panels):
            # Common terms for panel i
            alpha = self.uniform_flow.alpha
            beta_i = self.beta[i]
            x_ic = self.panels[i].xc
            y_ic = self.panels[i].yc

            for j in range(self.n_panels):
                # Diagonal terms
                if i == j:
                    an[i, j] = 0.5
                    at[i, j] = 0.0
                    continue

                # Geometric coefficients common to normal and tangential terms
                beta_j = self.beta[j]
                x_j1 = self.panels[j].x1
                y_j1 = self.panels[j].y1
                s_j = self.panels[j].l
                a1 = (x_j1 - x_ic) * np.sin(beta_j + alpha) + (y_ic - y_j1) * np.cos(
                    beta_j + alpha
                )
                a2 = (x_ic - x_j1) ** 2 + (y_ic - y_j1) ** 2
                a5 = (a2 - a1**2) ** 0.5

                # Normal influence coefficient
                a3n = np.sin(beta_i - beta_j)
                a4n = (x_ic - x_j1) * np.cos(beta_i + alpha) + (y_ic - y_j1) * np.sin(
                    beta_i + alpha
                )
                an[i, j] = SourcePanelSystem.integration_term(a1, a2, a3n, a4n, a5, s_j)

                # Tangential influence coefficient
                a3t = -np.cos(beta_j - beta_i)
                a4t = (x_ic - x_j1) * np.sin(beta_i + alpha) - (y_ic - y_j1) * np.cos(
                    beta_i + alpha
                )
                at[i, j] = SourcePanelSystem.integration_term(a1, a2, a3t, a4t, a5, s_j)

            # Freestream contributions to the RHS
            bn[i] = -self.uniform_flow.u_inf * np.cos(beta_i)
            bt[i] = -self.uniform_flow.u_inf * np.sin(beta_i)

        # Solve for panel strengths
        self.strengths = solve(an, bn)

        # Evaluate surface normal and tangential velocities and pressure coefficient
        self.vn = an @ self.strengths - bn
        self.vt = at @ self.strengths - bt
        self.cp = 1.0 - (self.vn**2 + self.vt**2) / self.uniform_flow.u_inf**2

    def velocity(self, x: np.ndarray, y: np.ndarray):
        """
        Computes the velocity field at arbitrary points in the flow domain.

        Parameters
        ----------
        x: 2D Numpy array of floats
            x-coordinates of the field points.
        y: 2D Numpy array of floats
            y-coordinates of the field points.

        Returns
        -------
        u: 2D Numpy array of floats
            x-component of the velocity field.
        v: 2D Numpy array of floats
            y-component of the velocity field.
        """
        # Flatten to 1D for matrix assembly, retaining original shape for output
        original_shape = x.shape
        x_flat = x.ravel()
        y_flat = y.ravel()

        # Preallocate influence coefficient and freestream RHS arrays
        n = x_flat.size
        ax, ay = np.zeros((n, self.n_panels)), np.zeros((n, self.n_panels))
        bx = np.full(n, -self.uniform_flow.u_inf * np.cos(self.uniform_flow.alpha))
        by = np.full(n, -self.uniform_flow.u_inf * np.sin(self.uniform_flow.alpha))

        # Populate influence coefficient matrices
        for i in range(n):
            # Common terms for field point i
            alpha = self.uniform_flow.alpha
            x_p = x_flat[i]
            y_p = y_flat[i]

            for j in range(self.n_panels):
                # Geometric coefficients common to x and y terms
                beta_j = self.beta[j]
                x_j1 = self.panels[j].x1
                y_j1 = self.panels[j].y1
                s_j = self.panels[j].l
                a1 = (x_j1 - x_p) * np.sin(beta_j + alpha) + (y_p - y_j1) * np.cos(
                    beta_j + alpha
                )
                a2 = (x_p - x_j1) ** 2 + (y_p - y_j1) ** 2
                a5 = (a2 - a1**2) ** 0.5

                # x-direction influence coefficient
                a3x = -np.sin(beta_j + alpha)
                a4x = x_p - x_j1
                ax[i, j] = SourcePanelSystem.integration_term(a1, a2, a3x, a4x, a5, s_j)

                # y-direction influence coefficient
                a3y = np.cos(beta_j + alpha)
                a4y = y_p - y_j1
                ay[i, j] = SourcePanelSystem.integration_term(a1, a2, a3y, a4y, a5, s_j)

        # Assemble velocity components and restore original array shape
        u = (ax @ self.strengths - bx).reshape(original_shape)
        v = (ay @ self.strengths - by).reshape(original_shape)

        return u, v
