import tkinter as tk
import numpy as np


class DrawingCanvas():

    def __init__(self, train_mode=False, model=None):
        # In case of creating dataset, set train mode to True
        self.train_mode = train_mode

        if not self.train_mode:
            if model is None:
                raise ValueError('Please define a model')
            self.model = model

        # Number of points for sampling gestures
        self.M = 40

        # File for saving dataset
        self.dataset_name = 'dataset_40.npz'

        # Mapping numbers to symbols
        self.symbol_map = {
            0: 'alpha',
            1: 'beta',
            2: 'gamma',
            3: 'delta',
            4: 'eta'
        }

        # Create window
        self.root = tk.Tk()

        # Sampled points from a gesture
        self.gesture_points = []

        # Drawing canvas
        self.canvas = tk.Canvas(self.root, width=400, height=400)

        if self.train_mode:
            # In case of train mode create label to track current symbol and number of samples
            self.current_symbol_label = tk.Label(
                self.canvas, text='alpha, number of samples: 0')
            self.current_symbol_label.config(font=("Courier", 44))
            self.X = []
            self.y = []
            self.current_label = [1, 0, 0, 0, 0]
            self.current_symbol_label.place(x=1, y=1)

            # Switching through classes
            for i in range(5):
                self.root.bind(str(i), self._assign_symbol)

            # Save dataset and end work
            self.root.bind('<Control-s>', self._write_dataset)

        else:
            self.predicted = tk.Label(self.canvas, text='Predicted: ')
            self.predicted.config(font=("Courier", 44))
            self.predicted.place(x=1, y=1)

        self.canvas.pack(fill="both", expand=True)
        self.canvas.old_coords = None

        self.root.bind('<B1-Motion>', self._motion)
        self.root.bind('<ButtonPress-1>', self._mouse_button)
        self.root.bind('<ButtonRelease-1>', self._mouse_button)
        self.root.mainloop()

    def _mouse_button(self, event):
        if str(event.type) == 'ButtonPress':
            self.canvas.delete('all')
            self.canvas.old_coords = event.x, event.y
            self.gesture_points = []

        elif str(event.type) == 'ButtonRelease':
            self.gesture_points = np.array(
                self.gesture_points).astype(np.float)
            self._vectorize()

            if self.train_mode:
                self.canvas.delete('all')
                self.X.append(self.gesture_points)
                self.y.append(self.current_label)
                self._update_label_text()
            else:
                X = np.array([self.gesture_points])
                preds = self.model.predict(X.reshape(
                    X.shape[0], X.shape[1] * X.shape[2]))
                pred_label = np.argmax(preds.ravel())
                self.predicted['text'] = "Predicted: {}".format(
                    self.symbol_map[pred_label])

    def _motion(self, event):
        x, y = event.x, event.y
        self.gesture_points.append([x, y])
        if self.canvas.old_coords:
            x1, y1 = self.canvas.old_coords
            self.canvas.create_line(x, y, x1, y1)
        self.canvas.old_coords = x, y

    def _assign_symbol(self, event):
        number_pressed = int(event.char)
        self.current_label = [0] * 5
        self.current_label[number_pressed] = 1
        self._update_label_text()

    def _update_label_text(self):
        current_symbol = int(np.argmax(self.current_label))
        self.current_symbol_label['text'] = "{}, number of samples: {}".format(
            self.symbol_map[current_symbol], int(np.sum(self.y, axis=0)[current_symbol]))

    def _write_dataset(self, event):

        np.savez(self.dataset_name, X=np.array(self.X), y=np.array(self.y))
        self.root.destroy()

    def _vectorize(self):
        """ Scales given gesture and samples it into M points
        """
        if len(self.gesture_points) == 0:
            return

        # Find centroid and center all samples
        self.gesture_points = self.gesture_points - \
            np.average(self.gesture_points, axis=0)

        # Scale to [-1, 1]
        m = max(np.max(np.abs(self.gesture_points), axis=0))
        self.gesture_points = self.gesture_points / m

        point_distances = np.sum([self.gesture_points[i + 1] - self.gesture_points[i]
                                  for i in range(len(self.gesture_points) - 1)], axis=0)
        point_distances_from_first = np.sqrt(np.sum(np.square([self.gesture_points[i] - self.gesture_points[0]
                                                               for i in range(len(self.gesture_points))]), axis=1))
        # Sample gesture
        gesture_len = np.sqrt(np.sum(np.square(point_distances)))

        sampled_points = []

        for distance in ([k * gesture_len/(self.M - 1) for k in range(self.M - 1)]):
            closest = self.gesture_points[np.abs(
                point_distances_from_first - distance).argmin()]

            sampled_points.append(closest)

        self.gesture_points = np.array(sampled_points)


def main():
    DrawingCanvas(train_mode=True)


if __name__ == "__main__":
    main()
