"""
    minimalistic Opengl based rendering tool

    date : 2017-20-03
"""
__author__ = "Mathieu Garon"
__version__ = "0.0.1"

import OpenGL.GL as gl
from vispy import app, gloo

from ulaval_6dof_object_tracking.utils.plyparser import PlyParser
import numpy as np
import os


class ModelRenderer(app.Canvas):
    def __init__(self, model_path, shader_path, camera, window_sizes, backend="pyglet"):
        """
        each window size will setup one fpo
        :param model_path:
        :param shader_path:
        :param camera:
        :param window_size: list of size tuple [(height, width), (height2, width2) ... ]
        """
        app.use_app(backend)  # Set backend
        app.Canvas.__init__(self, show=False, size=(camera.width, camera.height))

        fragment_code = open(os.path.join(shader_path, "fragment_light2.txt"), 'r').read()
        vertex_code = open(os.path.join(shader_path, "vertex_light2.txt"), 'r').read()

        model_3d = PlyParser(model_path)

        #gloo.gl.use_gl('gl2 debug')

        self.window_sizes = window_sizes
        self.camera = camera

        self.rgb = np.array([])
        self.depth = np.array([])

        # Create buffers
        vertices = model_3d.get_vertex()
        faces = model_3d.get_faces()
        self.data = np.ones(vertices.shape[0], [('a_position', np.float32, 3),
                                           ('a_color', np.float32, 3),
                                           ('a_normal', np.float32, 3),
                                           ('a_ambiant_occlusion', np.float32, 3),
                                           ('a_texcoords', np.float32, 2)])
        self.data['a_position'] = vertices
        self.data['a_color'] = model_3d.get_vertex_color()
        self.data['a_color'] /= 255.
        self.data['a_normal'] = model_3d.get_vertex_normals()
        self.data['a_normal'] = self.data['a_normal'] / np.linalg.norm(self.data['a_normal'], axis=1)[:, np.newaxis]

        # set white texture by default
        texture = np.ones((1, 1, 4), dtype=np.uint8)
        texture.fill(255)
        # else load the texture from the model
        try:
            self.data['a_texcoords'] = model_3d.get_texture_coord()
            texture = model_3d.get_texture()
            if texture is not None:
                texture = texture[::-1, :, :]
        except KeyError:
            pass

        self.vertex_buffer = gloo.VertexBuffer(self.data)
        self.index_buffer = gloo.IndexBuffer(faces.flatten().astype(np.uint32))

        self.program = gloo.Program(vertex_code, fragment_code)

        self.program.bind(self.vertex_buffer)
        self.program['tex'] = gloo.Texture2D(texture)

        self.program['shininess'] = [5.0]
        self.program['lightA_diffuse'] = [1, 1, 1]
        self.program['lightA_direction'] = [-1, -1, 1]
        self.program['lightA_specular'] = [1, 1, 1]

        self.program['ambientLightForce'] = [0.65, 0.65, 0.65]
        self.setup_camera(self.camera, 0, self.camera.width, self.camera.height, 0)

        # Frame buffer object
        self.fbos = []
        for window_size in self.window_sizes:
            shape = (window_size[1], window_size[0])
            self.fbos.append(gloo.FrameBuffer(gloo.Texture2D(shape=shape + (3,)), gloo.RenderBuffer(shape)))

    def load_ambiant_occlusion_map(self, path):
        try:
            ao_model = PlyParser(path)
            self.data["a_ambiant_occlusion"] = ao_model.get_vertex_color()
            self.data["a_ambiant_occlusion"] /= 255
        except FileNotFoundError:
            print("[WARNING] ViewpointRender: ambiant occlusion file not found ... continue with basic render")

    def setup_camera(self, camera, left, right, bottom, top):
        self.near_plane = 0.01
        self.far_plane = 2.5

        # credit : http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/

        proj = np.array([[camera.focal_x, 0, -camera.center_x, 0],
                         [0, camera.focal_y, -camera.center_y, 0],
                         [0, 0, self.near_plane + self.far_plane, self.near_plane * self.far_plane],
                         [0, 0, -1, 0]])
        self.projection_matrix = self.orthographicMatrix(left,
                                                         right,
                                                         bottom,
                                                         top,
                                                         self.near_plane,
                                                         self.far_plane).dot(proj).T
        self.program['proj'] = self.projection_matrix

    @staticmethod
    def orthographicMatrix(left, right, bottom, top, near, far):
        right = float(right)
        left = float(left)
        top = float(top)
        bottom = float(bottom)
        mat = np.array([[2. / (right - left), 0, 0, -(right + left) / (right - left)],
                        [0, 2. / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                        [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                        [0, 0, 0, 1]], dtype=np.float32)
        return mat

    def on_draw(self, event, fbo_index):
        size = self.window_sizes[fbo_index]
        fbo = self.fbos[fbo_index]
        with fbo:
            gloo.set_state(depth_test=True)
            #gl.glEnable(gl.GL_LINE_SMOOTH)
            #gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
            gloo.set_cull_face('back')  # Back-facing polygons will be culled
            gloo.clear(color=True, depth=True)
            gloo.set_viewport(0, 0, *size)
            self.program.draw('triangles', self.index_buffer)

            # Retrieve the contents of the FBO texture
            self.rgb = gl.glReadPixels(0, 0, size[0], size[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            self.rgb = np.frombuffer(self.rgb, dtype=np.uint8).reshape((size[1], size[0], 3))
            self.depth = gl.glReadPixels(0, 0, size[0], size[1], gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
            self.depth = self.depth.reshape(size[::-1])
            self.depth = self.gldepth_to_worlddepth(self.depth)

    def gldepth_to_worlddepth(self, frame):
        A = self.projection_matrix[2, 2]
        B = self.projection_matrix[3, 2]
        distance = B / (frame * -2.0 + 1.0 - A) * -1
        idx = distance[:, :] >= B / (A + 1)
        distance[idx] = 0
        return (distance * 1000).astype(np.uint16)

    def render_image(self, view_transform, fbo_index=0, light_direction=None, light_diffuse=None, ambiant_light=None):
        light_normal = np.ones(4)
        if light_direction is None:
            light_direction = np.array([0, 0.1, -0.9])
        if ambiant_light is None:
            ambiant_light = np.array([0.65, 0.65, 0.65])
        if light_diffuse is None:
            light_diffuse = np.array([0.4, 0.4, 0.4])

        light_normal[0:3] = light_direction
        light_direction = np.dot(view_transform.transpose().inverse().matrix, light_normal)[:3]
        self.program["lightA_direction"] = light_direction
        self.program["ambientLightForce"] = ambiant_light
        self.program['lightA_diffuse'] = light_diffuse
        self.program['view'] = view_transform.matrix.T

        self.update()
        self.on_draw(None, fbo_index)
        return self.rgb, self.depth