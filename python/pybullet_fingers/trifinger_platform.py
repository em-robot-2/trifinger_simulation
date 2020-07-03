import numpy as np

from .tasks import move_cube
from .sim_finger import SimFinger
from . import camera, collision_objects


class ObjectPose:
    """A pure-python copy of trifinger_object_tracking::ObjectPose."""

    __slots__ = ["position", "orientation", "timestamp", "confidence"]

    def __init__(self):
        #: array: Position (x, y, z) of the object.  Units are meters.
        self.position = np.zeros(3)
        #: array: Orientation of the object as (x, y, z, w) quaternion.
        self.orientation = np.zeros(4)
        #: float: Timestamp when the pose was observed.
        self.timestamp = 0.0
        #: float: Estimate of the confidence for this pose observation.
        self.confidence = 1.0


class CameraObservation:
    """Pure-python copy of trifinger_cameras.camera.CameraObservation."""

    __slots__ = ["image", "timestamp"]

    def __init__(self):
        #: array: The image.
        self.image = None
        #: float: Timestamp when the image was received.
        self.timestamp = None


class TriCameraObservation:
    """Pure-python copy of trifinger_cameras.tricamera.TriCameraObservation."""

    __slots__ = ["cameras"]

    def __init__(self):
        #: list of :class:`CameraObservation`: List of observations of cameras
        #: "camera60", "camera180" and "camera300" (in this order).
        self.cameras = [CameraObservation() for i in range(3)]


class TriFingerPlatform:
    """
    Wrapper around the simulation providing the same interface as
    ``robot_interfaces::TriFingerPlatformFrontend``.

    The following methods of the robot_interfaces counterpart are not
    supported:

    - get_robot_status()
    - wait_until_timeindex()

    """

    def __init__(self, visualization=False, initial_object_pose=None):
        """Initialize.

        Args:
            visualization (bool):  Set to true to run visualization.
            initial_object_pose:  Initial position for the manipulation object.
                Tuple with position (x, y, z) and orientation quaternion
                (x, y, z, w).  This is optional, if not set, a random pose will
                be sampled.
        """
        # Initially move the fingers to a pose where they are guaranteed to not
        # collide with the object on the ground.
        initial_position = [0.0, np.deg2rad(-70), np.deg2rad(-130)] * 3

        self.simfinger = SimFinger(0.001, visualization, "trifingerone")

        # set fingers to initial pose
        self.simfinger.reset_finger(initial_position)

        if initial_object_pose is None:
            initial_object_pose = move_cube.sample_goal(difficulty=-1)
        self.cube = collision_objects.Block(*initial_object_pose)

        self.tricamera = camera.TriFingerCameras()

        # forward "RobotFrontend" methods directly to simfinger
        self.Action = self.simfinger.Action
        self.append_desired_action = self.simfinger.append_desired_action
        self.get_desired_action = self.simfinger.get_desired_action
        self.get_applied_action = self.simfinger.get_applied_action
        self.get_timestamp_ms = self.simfinger.get_timestamp_ms
        self.get_current_timeindex = self.simfinger.get_current_timeindex
        self.get_robot_observation = self.simfinger.get_observation

    def get_object_pose(self, t):
        """Get object pose at time step t.

        Args:
            t:  The time index of the step for which the object pose is
                requested.  Only the value returned by the last call of
                :meth:`~append_desired_action` is valid.

        Returns:
            ObjectPose:  Pose of the object.  Values come directly from the
            simulation without adding noise, so the confidence is 1.0.

        Raises:
            ValueError: If invalid time index ``t`` is passed.
        """
        self.simfinger._validate_time_index(t)

        cube_state = self.cube.get_state()
        pose = ObjectPose()
        pose.position = np.asarray(cube_state[0])
        pose.orientation = np.asarray(cube_state[1])
        pose.timestamp = self.get_timestamp_ms(t) * 1000.0
        pose.confidence = 1.0

        return pose

    def get_camera_observation(self, t):
        """Get camera observation at time step t.

        Args:
            t:  The time index of the step for which the observation is
                requested.  Only the value returned by the last call of
                :meth:`~append_desired_action` is valid.

        Returns:
            TriCameraObservation:  Observations of the three cameras.  Images
            are rendered in the simulation.  Note that they are not optimized
            to look realistically.

        Raises:
            ValueError: If invalid time index ``t`` is passed.
        """
        self.simfinger._validate_time_index(t)

        images = self.tricamera.get_images()
        timestamp = self.get_timestamp_ms(t) * 1000.0

        observation = TriCameraObservation()
        for i, image in enumerate(images):
            observation.cameras[i].image = image
            observation.cameras[i].timestamp = timestamp

        return observation