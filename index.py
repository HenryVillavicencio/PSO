
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import funcnames
import numpy as np
from tkinter import *
from tkinter import ttk
from matplotlib import cm
import time

matplotlib.use('TkAgg')

switcher_func = {
    1: funcnames.Ackley,
    2: funcnames.SCB,
    3: funcnames.DeJong_F1,
    4: funcnames.DeJong_F2,
    5: funcnames.Rastrigin,
    6: funcnames.Schaffer_F6,
    7: funcnames.Peaks,
    8: funcnames.Passino,
    9: funcnames.Schewefel
}


class Gui:
    def restart_plot(self):
        switcher = {
            1: {
                'lim_supx': 20,
                'lim_infx': -20,
                'lim_supy': 20,
                'lim_infy': -20,
                'lim_supz': 15,
                'lim_infz': 0
            },
            2: {
                'lim_supx': 20,
                'lim_infx': -20,
                'lim_supy': 20,
                'lim_infy': -20,
                'lim_supz': 2.5e7,
                'lim_infz': -0.5
            },
            3: {
                'lim_supx': 20,
                'lim_infx': -20,
                'lim_supy': 20,
                'lim_infy': -20,
                'lim_supz': 800,
                'lim_infz': 0
            },
            4: {
                'lim_supx': 20,
                'lim_infx': -20,
                'lim_supy': 20,
                'lim_infy': -20,
                'lim_supz': 2.5e7,
                'lim_infz': 0
            },
            5: {
                'lim_supx': 20,
                'lim_infx': -20,
                'lim_supy': 20,
                'lim_infy': -20,
                'lim_supz': 800,
                'lim_infz': 0
            },
            6: {
                'lim_supx': 20,
                'lim_infx': -20,
                'lim_supy': 20,
                'lim_infy': -20,
                'lim_supz': 1,
                'lim_infz': 0
            },
            7: {
                'lim_supx': 3,
                'lim_infx': -3,
                'lim_supy': 3,
                'lim_infy': -3,
                'lim_supz': 10,
                'lim_infz': -10
            },
            8: {
                'lim_supx': 30,
                'lim_infx': 0,
                'lim_supy': 30,
                'lim_infy': 0,
                'lim_supz': 5,
                'lim_infz': -5
            },
            9: {
                'lim_supx': 40,
                'lim_infx': -40,
                'lim_supy': 40,
                'lim_infy': -40,
                'lim_supz': 40,
                'lim_infz': -40
            }
        }
        lims = switcher.get(self.index)
        Nx = 50
        Ny = 50
        self.Rx = [lims['lim_infx'], lims['lim_supx']]
        self.Ry = [lims['lim_infy'], lims['lim_supy']]
        self.Rz = [lims['lim_infz'], lims['lim_supz']]
        stepx = sum([abs(i) for i in self.Rx])/(Nx-1)
        stepy = sum([abs(i) for i in self.Ry])/(Ny-1)
        self.theta1 = np.repeat(
            [np.arange(self.Rx[0], self.Rx[1]+stepx/2, stepx)], Ny, axis=0)
        self.theta2 = np.repeat(
            [np.arange(self.Ry[0], self.Ry[1]+stepy/2, stepy)], Nx, axis=0).T

        theta = np.concatenate(
            (np.array([np.concatenate(self.theta1.T, axis=0)]), np.array([np.concatenate(self.theta2.T, axis=0)])), axis=0)

        func = switcher_func.get(self.index)
        self.j = func(theta)
        self.j = self.j.reshape(Ny, Nx).T

        self.ax.clear()
        self.ax.plot_surface(self.theta1, self.theta2, self.j, rstride=1,
                             cstride=1, cmap='terrain', alpha=0.7)
        self.canvas.draw()

        self.ax2.clear()
        self.ax2.contour(self.theta1, self.theta2, self.j, cmap=cm.coolwarm)
        self.canvas2.draw()

        self.ax3.clear()
        self.canvas3.draw()

    def onChange_menu(self, event):
        self.index = self.OPTIONS.index(event)
        self.index = self.index + 1
        self.restart_plot()

    def onClick(self):
        # % Parametros Iniciales
        iteraciones = int(self.entry_iter.get())  # % Numero de iteraciones
        iter = 0  # % Contador de iteracciones
        # % Parametros PSO
        w = 0.4
        c1 = 2
        c2 = 2
        Vmax = self.Rx[1]
        Part = int(self.entry_part.get())  # % Numero de particulas

        D = 2  # % Dimension del espacio
        Lim_inf = self.Rx[0]  # % inicial de las particulas.
        # % Limites empleados para la establecer la posicion
        Lim_sup = self.Rx[1]
        # % X y V son de tamano Part x D.
        # % Vectores fila
        X = np.random.rand(Part, D) * (Lim_sup-Lim_inf) + Lim_inf
        V = np.random.rand(Part, D)
        XT = X.T
        func = switcher_func.get(self.index)

        val_func = func(XT)

        self.ax.scatter(XT[0], XT[1], val_func, c='r',  marker='*')
        self.canvas.draw()

        self.ax2.scatter(XT[0], XT[1], c='r', marker='*')
        self.canvas2.draw()

        # %Funcion Objetivo a Optimizar
        # % Evalua la funcion objetivo.

        # % Xi es la mejor posicion de cada una de las particula
        Xi = np.copy(X)
        # % val_Xi el correspondiente valor al evaluar la func_obj.
        val_Xi = np.copy(val_func)

        # % Encuentra la mejor particula de la pobalcion inicial
        # % val_G es el mejor valor y g la posicion relativa de la particula
        val_G = np.amin(val_func)
        g = np.argmin(val_func)
        Xg = X[g]
        fxmin = val_G
        Xg_t = np.repeat([Xg], Part, axis=0)

        while iter <= iteraciones:
            self.label_iter.set(iter)
            self.label_error.set(val_G)
            iter = iter+1
            R1 = abs(np.random.rand(Part, D))
            R2 = abs(np.random.rand(Part, D))
            #  % Calculo de la velocidad

            V = (w*V)+((c1*R1)*(Xi-X))+((c2*R2)*(Xg_t-X))  # %PSO
            V[V > Vmax] = Vmax
            V[V < -Vmax] = -Vmax
            X = X + V
            XT = X.T
            val_func = func(XT)

            self.ax.clear()
            self.ax.plot_surface(self.theta1, self.theta2, self.j, rstride=1,
                                 cstride=1, cmap='terrain', alpha=0.7)
            self.ax.scatter(XT[0], XT[1], val_func, c='r',  marker='*')
            self.canvas.draw()

            self.ax2.clear()
            self.ax2.contour(self.theta1, self.theta2,
                             self.j, cmap=cm.coolwarm)
            self.ax2.scatter(XT[0], XT[1], c='r', marker='*')
            self.canvas2.draw()

            self.ax3.plot([(iter-1), iter], [fxmin, val_G])
            self.canvas3.draw()

            fxmin = np.amin(val_Xi)
            b = np.argmin(val_Xi)
            xmin = Xi[b]
            self.label_position.set(xmin)

            for k in range(Part):
                if(val_func[k] < val_Xi[k]):
                    val_Xi[k] = val_func[k]
                    Xi[k] = X[k]
                if(val_func[k] < val_G):
                    val_G = val_func[k]
                    Xg = X[k]
                    Xg_t = np.repeat([Xg], Part, axis=0)

            self.wind.update()
            time.sleep(0.3)

    def __init__(self, window):
        self.wind = window
        self.wind.title('POS')

        self.index = 1

        fig = plt.figure(figsize=(5, 5), dpi=80)
        self.ax = fig.gca(projection='3d')
        self.ax.view_init(30, 240)
        self.canvas = FigureCanvasTkAgg(fig, master=self.wind)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        self.canvas.draw()

        fig2 = plt.figure(figsize=(4, 4), dpi=80)
        self.ax2 = fig2.subplots()
        self.canvas2 = FigureCanvasTkAgg(fig2, master=self.wind)
        self.canvas2.get_tk_widget().grid(row=1, column=1)
        self.canvas2.draw()

        frame_plot = Frame(self.wind)
        fig3 = plt.figure(figsize=(4, 4), dpi=80)
        self.ax3 = fig3.subplots()
        self.canvas3 = FigureCanvasTkAgg(fig3, master=frame_plot)
        self.canvas3.get_tk_widget().grid(row=0, column=0)
        self.canvas3.draw()

        self.label_error = StringVar()
        Label(frame_plot, textvariable=self.label_error).grid(row=1, column=0)

        self.label_iter = StringVar()
        Label(frame_plot, textvariable=self.label_iter).grid(row=2, column=0)

        frame_plot.grid(row=0, column=1)

        frame = Frame(self.wind)
        Label(frame, text="Número de partículas").grid(row=0, sticky=W)
        Label(frame, text="Iteraciones").grid(row=1, sticky=W)
        Label(frame, text="Función").grid(row=2, sticky=W)
        Label(frame, text="Posición").grid(row=4, sticky=W)

        self.label_position = StringVar()
        Label(frame, textvariable=self.label_position).grid(row=4, column=1)

        self.entry_part = Entry(frame)
        self.entry_part.insert(END, '10')

        self.entry_iter = Entry(frame)
        self.entry_iter.insert(END, '100')

        self.entry_part.grid(row=0, column=1)
        self.entry_iter.grid(row=1, column=1)

        self.OPTIONS = [
            "Ackley",
            "SCB",
            "DeJong F1",
            "DeJong F2",
            "Rastrigin",
            "Schaffer F6",
            "Peaks",
            "Passino",
            "Schewefel",
        ]
        variable = StringVar(frame)
        variable.set(self.OPTIONS[0])
        menu = OptionMenu(* (frame, variable) +
                          tuple(self.OPTIONS), command=self.onChange_menu)
        menu.grid(row=2, column=1)

        button = Button(frame, text="Iniciar",
                        command=self.onClick, width=20)
        button.grid(row=5, column=0)
        frame.grid(row=1, column=0)
        self.restart_plot()


if __name__ == "__main__":

    window = Tk()

    def _quit():
        window.quit()
        window.destroy()

    aplication = Gui(window)
    window.protocol("WM_DELETE_WINDOW",  _quit)
    window.mainloop()
