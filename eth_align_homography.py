import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.widgets import Slider, Button
from matplotlib.transforms import Affine2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_raw_data(file_path, delim='\t'):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line_data = []
            for i in line.strip().split(delim):
                try:
                    line_data.append(float(i))
                except ValueError:
                    pass
            if line_data:
                data.append(line_data)
    return np.asarray(data)

def load_homography_matrix(h_path):
    # Accept standard H.txt files saved with tabs or generic whitespace,
    # including scientific notation values like 1.23e-04.
    H = np.loadtxt(h_path)
    H = np.asarray(H, dtype=np.float32)
    if H.shape != (3, 3):
        raise ValueError(f'Expected a 3x3 homography matrix in {h_path}, got shape {H.shape}.')
    return H

def build_initial_overlay_params(world_coords, image_shape):
    x_coords = world_coords[:, 0]
    y_coords = world_coords[:, 1]
    center_x = float((x_coords.min() + x_coords.max()) / 2.0)
    center_y = float((y_coords.min() + y_coords.max()) / 2.0)
    span_x = float(max(x_coords.max() - x_coords.min(), 1.0))
    span_y = float(max(y_coords.max() - y_coords.min(), 1.0))
    img_h, img_w = image_shape[:2]
    scale = max(span_x / img_w, span_y / img_h)
    return center_x, center_y, scale

def image_transform(center_x, center_y, scale, angle_deg, image_shape, flip_x=False, flip_y=False):
    img_h, img_w = image_shape[:2]
    scale_x = -scale if flip_x else scale
    scale_y = -scale if flip_y else scale
    return (
        Affine2D()
        .translate(-img_w / 2.0, -img_h / 2.0)
        .scale(scale_x, scale_y)
        .rotate_deg(angle_deg)
        .translate(center_x, center_y)
    )

def image_corner_pixels(image_shape):
    img_h, img_w = image_shape[:2]
    return np.array([
        [0.0, 0.0],
        [img_w - 1.0, 0.0],
        [img_w - 1.0, img_h - 1.0],
        [0.0, img_h - 1.0],
    ], dtype=np.float32)

def overlay_world_corners(overlay_params, image_shape):
    transform = image_transform(
        overlay_params['center_x'],
        overlay_params['center_y'],
        overlay_params['scale'],
        overlay_params['angle_deg'],
        image_shape,
        flip_x=overlay_params.get('flip_x', False),
        flip_y=overlay_params.get('flip_y', False),
    )
    return np.asarray(transform.transform(image_corner_pixels(image_shape)), dtype=np.float32)

def overlay_image_alignment(world_coords, img):
    center_x, center_y, scale = build_initial_overlay_params(world_coords, img.shape)
    x_margin = max(np.ptp(world_coords[:, 0]) * 0.1, 1.0)
    y_margin = max(np.ptp(world_coords[:, 1]) * 0.1, 1.0)

    fig, ax = plt.subplots(figsize=(11, 9))
    plt.subplots_adjust(left=0.1, bottom=0.28)

    ax.plot(world_coords[:, 0], world_coords[:, 1], 'b.', markersize=1, alpha=0.45)
    image_artist = ax.imshow(img, origin='upper', alpha=0.45)
    ax.set_title(
        'Align image overlay to trajectories, then click Use Current Overlay\n'
        'Adjust center, scale, rotation, and alpha for a precise fit'
    )
    ax.set_xlabel('Meters X')
    ax.set_ylabel('Meters Y')
    ax.grid(True)
    ax.set_xlim(world_coords[:, 0].min() - x_margin, world_coords[:, 0].max() + x_margin)
    ax.set_ylim(world_coords[:, 1].min() - y_margin, world_coords[:, 1].max() + y_margin)
    ax.set_aspect('equal', adjustable='box')

    def apply_transform(val=None):
        transform = image_transform(
            center_x_slider.val,
            center_y_slider.val,
            scale_slider.val,
            angle_slider.val,
            img.shape,
            flip_x=alignment['flip_x'],
            flip_y=alignment['flip_y'],
        )
        image_artist.set_transform(transform + ax.transData)
        image_artist.set_alpha(alpha_slider.val)
        fig.canvas.draw_idle()

    ax_center_x = plt.axes((0.14, 0.18, 0.68, 0.03))
    ax_center_y = plt.axes((0.14, 0.14, 0.68, 0.03))
    ax_scale = plt.axes((0.14, 0.10, 0.68, 0.03))
    ax_angle = plt.axes((0.14, 0.06, 0.68, 0.03))
    ax_alpha = plt.axes((0.14, 0.02, 0.68, 0.03))
    ax_flip_x = plt.axes((0.84, 0.20, 0.12, 0.07))
    ax_flip_y = plt.axes((0.84, 0.11, 0.12, 0.07))
    ax_identity_button = plt.axes((0.84, 0.02, 0.12, 0.07))

    center_x_slider = Slider(ax_center_x, 'Center X', world_coords[:, 0].min(), world_coords[:, 0].max(), valinit=center_x)
    center_y_slider = Slider(ax_center_y, 'Center Y', world_coords[:, 1].min(), world_coords[:, 1].max(), valinit=center_y)
    scale_slider = Slider(ax_scale, 'Scale', scale * 0.1, scale * 5.0, valinit=scale)
    angle_slider = Slider(ax_angle, 'Rotation', -180.0, 180.0, valinit=0.0)
    alpha_slider = Slider(ax_alpha, 'Alpha', 0.05, 1.0, valinit=0.45)
    confirm_button = Button(plt.axes((0.84, 0.29, 0.12, 0.07)), 'Use Current\nOverlay')
    flip_x_button = Button(ax_flip_x, 'Flip X\nOFF')
    flip_y_button = Button(ax_flip_y, 'Flip Y\nOFF')
    identity_button = Button(ax_identity_button, 'Already\nAligned')
    ax_flip_x.set_title('OFF', fontsize=8, pad=1)
    ax_flip_y.set_title('OFF', fontsize=8, pad=1)

    alignment = {'confirmed': False, 'identity': False, 'flip_x': False, 'flip_y': False}

    def confirm(event):
        alignment['confirmed'] = True
        plt.close(fig)

    def use_identity(event):
        alignment['confirmed'] = True
        alignment['identity'] = True
        plt.close(fig)

    def toggle_flip_x(event):
        alignment['flip_x'] = not alignment['flip_x']
        ax_flip_x.set_title('ON' if alignment['flip_x'] else 'OFF', fontsize=8, pad=1)
        apply_transform()

    def toggle_flip_y(event):
        alignment['flip_y'] = not alignment['flip_y']
        ax_flip_y.set_title('ON' if alignment['flip_y'] else 'OFF', fontsize=8, pad=1)
        apply_transform()

    for slider in [center_x_slider, center_y_slider, scale_slider, angle_slider, alpha_slider]:
        slider.on_changed(apply_transform)
    confirm_button.on_clicked(confirm)
    flip_x_button.on_clicked(toggle_flip_x)
    flip_y_button.on_clicked(toggle_flip_y)
    identity_button.on_clicked(use_identity)

    apply_transform()
    plt.show()

    if not alignment['confirmed']:
        raise RuntimeError('Overlay alignment was cancelled.')

    return {
        'center_x': center_x_slider.val,
        'center_y': center_y_slider.val,
        'scale': scale_slider.val,
        'angle_deg': angle_slider.val,
        'alpha': alpha_slider.val,
        'use_identity': alignment['identity'],
        'flip_x': alignment['flip_x'],
        'flip_y': alignment['flip_y'],
        'xlim': ax.get_xlim(),
        'ylim': ax.get_ylim(),
    }

def show_verification(world_coords, img, overlay_params, H):
    fig, ax = plt.subplots(figsize=(10, 8))

    if overlay_params['use_identity']:
        image_artist = ax.imshow(img, origin='upper', alpha=overlay_params['alpha'])
        transform = image_transform(
            overlay_params['center_x'],
            overlay_params['center_y'],
            overlay_params['scale'],
            overlay_params['angle_deg'],
            img.shape,
            flip_x=overlay_params.get('flip_x', False),
            flip_y=overlay_params.get('flip_y', False),
        )
        image_artist.set_transform(transform + ax.transData)
        ax.plot(world_coords[:, 0], world_coords[:, 1], 'r.', markersize=1, alpha=0.3)
        ax.set_xlim(overlay_params['xlim'])
        ax.set_ylim(overlay_params['ylim'])
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Verification: Identity Homography Preserves Current Alignment')
        ax.set_xlabel('Meters X')
        ax.set_ylabel('Meters Y')
        ax.grid(True)
    else:
        world_pts_homogeneous = np.hstack([world_coords, np.ones((world_coords.shape[0], 1))])
        projected_pts = (H @ world_pts_homogeneous.T).T
        projected_pts[:, 0] /= projected_pts[:, 2]
        projected_pts[:, 1] /= projected_pts[:, 2]
        ax.imshow(img)
        ax.plot(projected_pts[:, 0], projected_pts[:, 1], 'r.', markersize=1, alpha=0.3)
        ax.set_title('Verification: Trajectories Projected onto Map')

    plt.show()

def preview_existing_homography(txt_path, img_path, h_path):
    print(f"Previewing existing homography from {h_path}...")
    img = plt.imread(img_path)
    raw_data = read_raw_data(txt_path)
    world_coords = raw_data[:, [2, 3]]
    H = load_homography_matrix(h_path)
    overlay_params = {
        'use_identity': False,
        'alpha': 1.0,
    }
    show_verification(world_coords, img, overlay_params, H)

def run_alignment_tool(txt_path, img_path, output_dir):
    print(f"Loading data from {txt_path} and {img_path}...")
    
    # Load Image
    img = plt.imread(img_path)
    
    # Load Trajectories
    raw_data = read_raw_data(txt_path)
    # Format is frame_id, agent_id, pos_x, pos_y
    world_coords = raw_data[:, [2, 3]] 
    overlay_params = overlay_image_alignment(world_coords, img)
    if overlay_params['use_identity']:
        print('Using identity homography. No geometric alteration will be applied.')
        H = np.eye(3, dtype=np.float32)
    else:
        world_pts = overlay_world_corners(overlay_params, img.shape)
        pixel_pts = image_corner_pixels(img.shape)

        print(f"World Corner Points Derived From Current Overlay: {world_pts}")
        
        # Step 3: Calculate Homography
        print("Calculating Homography Matrix...")
        H, status = cv2.findHomography(world_pts, pixel_pts)
    
    print("Calculated H Matrix:")
    print(H)
    
    # Step 4: Save to file
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'H.txt')
    np.savetxt(save_path, H, delimiter='\t')
    print(f"Saved homography matrix to {save_path}")
    
    # Step 5: Visual Verification
    print("Testing projection...")
    show_verification(world_coords, img, overlay_params, H)

class AlignmentToolGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ETH Homography Alignment Tool")
        self.root.resizable(False, False)

        self.trajectory_path = tk.StringVar()
        self.image_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.homography_path = tk.StringVar()

        self._build()

    def _build(self):
        frame = tk.Frame(self.root, padx=12, pady=12)
        frame.pack(fill='both', expand=True)

        tk.Label(frame, text="Trajectory File").grid(row=0, column=0, sticky='w', pady=(0, 6))
        tk.Entry(frame, textvariable=self.trajectory_path, width=55).grid(row=1, column=0, padx=(0, 8), sticky='we')
        tk.Button(frame, text="Browse", command=self.select_trajectory_file, width=10).grid(row=1, column=1, sticky='e')

        tk.Label(frame, text="Image File").grid(row=2, column=0, sticky='w', pady=(12, 6))
        tk.Entry(frame, textvariable=self.image_path, width=55).grid(row=3, column=0, padx=(0, 8), sticky='we')
        tk.Button(frame, text="Browse", command=self.select_image_file, width=10).grid(row=3, column=1, sticky='e')

        tk.Label(frame, text="Output Directory").grid(row=4, column=0, sticky='w', pady=(12, 6))
        tk.Entry(frame, textvariable=self.output_dir, width=55).grid(row=5, column=0, padx=(0, 8), sticky='we')
        tk.Button(frame, text="Browse", command=self.select_output_dir, width=10).grid(row=5, column=1, sticky='e')

        tk.Label(frame, text="Existing H.txt (Optional Preview)").grid(row=6, column=0, sticky='w', pady=(12, 6))
        tk.Entry(frame, textvariable=self.homography_path, width=55).grid(row=7, column=0, padx=(0, 8), sticky='we')
        tk.Button(frame, text="Browse", command=self.select_homography_file, width=10).grid(row=7, column=1, sticky='e')

        tk.Button(frame, text="Preview Or Align", command=self.run, width=18).grid(row=8, column=0, columnspan=2, pady=(16, 0))

    def select_trajectory_file(self):
        path = filedialog.askopenfilename(
            title="Select Trajectory File",
            initialdir=BASE_DIR,
            filetypes=[("Text Files", "*.txt *.csv"), ("All Files", "*.*")]
        )
        if path:
            self.trajectory_path.set(path)

    def select_image_file(self):
        path = filedialog.askopenfilename(
            title="Select Background Image",
            initialdir=BASE_DIR,
            filetypes=[("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")]
        )
        if path:
            self.image_path.set(path)

    def select_output_dir(self):
        path = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=BASE_DIR
        )
        if path:
            self.output_dir.set(path)

    def select_homography_file(self):
        path = filedialog.askopenfilename(
            title="Select Existing Homography File",
            initialdir=BASE_DIR,
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if path:
            self.homography_path.set(path)

    def run(self):
        txt_path = self.trajectory_path.get().strip()
        img_path = self.image_path.get().strip()
        output_dir = self.output_dir.get().strip()
        h_path = self.homography_path.get().strip()

        if not txt_path or not os.path.isfile(txt_path):
            messagebox.showerror("Missing Trajectory File", "Select a valid trajectory file.")
            return

        if not img_path or not os.path.isfile(img_path):
            messagebox.showerror("Missing Image File", "Select a valid background image file.")
            return

        if h_path:
            if not os.path.isfile(h_path):
                messagebox.showerror("Missing Homography File", "Select a valid existing H.txt file.")
                return

            try:
                preview_existing_homography(txt_path, img_path, h_path)
            except Exception as exc:
                messagebox.showerror("Preview Failed", str(exc))
                return

            messagebox.showinfo("Preview Complete", f"Previewed homography from {h_path}")
            return

        if not output_dir:
            messagebox.showerror("Missing Output Directory", "Select an output directory.")
            return

        try:
            run_alignment_tool(txt_path, img_path, output_dir)
        except Exception as exc:
            messagebox.showerror("Alignment Failed", str(exc))
            return

        messagebox.showinfo("Done", f"Saved homography matrix to {os.path.join(output_dir, 'H.txt')}")

def launch_gui():
    root = tk.Tk()
    AlignmentToolGUI(root)
    root.mainloop()

if __name__ == '__main__':
    launch_gui()