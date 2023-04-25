import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil

def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            source_scale = 0
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                outputs[("position_depth", scale, frame_id)] = self.position_depth[source_scale](
                        cam_points, inputs[("K", source_scale)], T)


def visualize_depth(depth, color=True):
    if color == 'color':
        # Saving colormapped depth image
        vmin = depth.min()
        vmax = np.percentile(depth, 99)
        normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
        colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        
    else:
        # Saving grayscale depth image
        im_depth = depth.astype(np.uint16)
        im = pil.fromarray(im_depth)
    return im 