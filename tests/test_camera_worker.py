from unittest.mock import MagicMock, patch

from reachy_mini_conversation_app.camera_worker import CameraWorker


def test_simulation_webcam_opens_successfully() -> None:
    """CameraWorker should store the capture object when webcam is available in simulation."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    with patch("cv2.VideoCapture", return_value=mock_cap):
        worker = CameraWorker(MagicMock(), is_simulation=True)

    assert worker._webcam is mock_cap


def test_simulation_webcam_fails_to_open() -> None:
    """CameraWorker should continue without camera when webcam cannot be opened."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        worker = CameraWorker(MagicMock(), is_simulation=True)

    assert worker._webcam is None
    mock_cap.release.assert_called_once()


def test_simulation_cv2_not_installed() -> None:
    """CameraWorker should continue without camera when opencv is not installed."""
    with patch.dict("sys.modules", {"cv2": None}):
        worker = CameraWorker(MagicMock(), is_simulation=True)

    assert worker._webcam is None


def test_real_robot_uses_media_get_frame() -> None:
    """CameraWorker should read frames from the robot camera when not in simulation mode."""
    mock_robot = MagicMock()
    worker = CameraWorker(mock_robot, is_simulation=False)

    def get_frame_and_stop():
        worker._stop_event.set()
        return None

    mock_robot.media.get_frame.side_effect = get_frame_and_stop

    with patch("time.sleep"):
        worker.working_loop()

    mock_robot.media.get_frame.assert_called_once()
