import grpc
import internal_communication_pb2
import internal_communication_pb2_grpc


def run_mission_process():
    channel = grpc.insecure_channel("localhost:50051")
    stub = internal_communication_pb2_grpc.InternalServiceStub(channel)

    # camera action: 0 - None, 1 - TakePhoto, 2 - StartPhotoInterval, 3 - StopPhotoInterval, 4 - StartVideo, 5 - StopVideo,
    # 6 - StartPhotoDistance, 7 - StopPhotoDistance

    # vehicle action: 0 - None, 1 - Takeoff, 2 - Land, 3 - TransitionToFw, 4 - TransitionToMc

    try:
        prompt_response = stub.GetPrompt(internal_communication_pb2.Empty())
        print(f"Prompt from C++: {prompt_response.prompt}")

        telemetry = stub.GetTelemetry(internal_communication_pb2.Empty())

        base_lat = telemetry.latitude_deg
        base_lon = telemetry.longitude_deg

        m10 = 0.00009  # 10 meters step

        item1 = internal_communication_pb2.MissionItem()
        item1.latitude_deg = base_lat
        item1.longitude_deg = base_lon
        item1.relative_altitude_m = 10.0
        item1.speed_m_s = 1.0
        item1.is_fly_through = False
        item1.gimbal_pitch_deg = float("nan")
        item1.gimbal_yaw_deg = float("nan")
        item1.camera_action = 0
        item1.loiter_time_s = float("nan")
        item1.camera_photo_interval_s = 0.1
        item1.acceptance_radius_m = 0.5
        item1.yaw_deg = 0.0
        item1.camera_photo_distance_m = float("nan")
        item1.vehicle_action = 1

        item2 = internal_communication_pb2.MissionItem()
        item2.latitude_deg = base_lat + m10
        item2.longitude_deg = base_lon
        item2.relative_altitude_m = 10.0
        item2.speed_m_s = 1.0
        item2.is_fly_through = False
        item2.gimbal_pitch_deg = float("nan")
        item2.gimbal_yaw_deg = float("nan")
        item2.camera_action = 0
        item2.loiter_time_s = 1.0
        item2.camera_photo_interval_s = 0.1
        item2.acceptance_radius_m = 0.5
        item2.yaw_deg = 0.0
        item2.camera_photo_distance_m = float("nan")
        item2.vehicle_action = 0

        item3 = internal_communication_pb2.MissionItem()
        item3.latitude_deg = base_lat + (m10 * 2)
        item3.longitude_deg = base_lon
        item3.relative_altitude_m = 10.0
        item3.speed_m_s = 1.0
        item3.is_fly_through = False
        item3.gimbal_pitch_deg = float("nan")
        item3.gimbal_yaw_deg = float("nan")
        item3.camera_action = 0
        item3.loiter_time_s = 1.0
        item3.camera_photo_interval_s = 0.1
        item3.acceptance_radius_m = 0.5
        item3.yaw_deg = 0.0
        item3.camera_photo_distance_m = float("nan")
        item3.vehicle_action = 0

        item4 = internal_communication_pb2.MissionItem()
        item4.latitude_deg = base_lat + m10
        item4.longitude_deg = base_lon
        item4.relative_altitude_m = 10.0
        item4.speed_m_s = 1.0
        item4.is_fly_through = True
        item4.gimbal_pitch_deg = float("nan")
        item4.gimbal_yaw_deg = float("nan")
        item4.camera_action = 0
        item4.loiter_time_s = float("nan")
        item4.camera_photo_interval_s = 0.1
        item4.acceptance_radius_m = 0.5
        item4.yaw_deg = 180.0
        item4.camera_photo_distance_m = float("nan")
        item4.vehicle_action = 0

        item5 = internal_communication_pb2.MissionItem()
        item5.latitude_deg = base_lat
        item5.longitude_deg = base_lon
        item5.relative_altitude_m = 10.0
        item5.speed_m_s = 1.0
        item5.is_fly_through = False
        item5.gimbal_pitch_deg = float("nan")
        item5.gimbal_yaw_deg = float("nan")
        item5.camera_action = 0
        item5.loiter_time_s = float("nan")
        item5.camera_photo_interval_s = 0.1
        item5.acceptance_radius_m = 0.5
        item5.yaw_deg = 180.0
        item5.camera_photo_distance_m = float("nan")
        item5.vehicle_action = 2

        mission_request = internal_communication_pb2.MissionItemList()
        mission_request.items.extend([item1, item2, item3, item4, item5])

        stub.StartMission(mission_request)
        print("Mission sent successfully")

    except grpc.RpcError as e:
        print(f"Error: {e.code()}")
    finally:
        channel.close()


if __name__ == "__main__":
    run_mission_process()
