import grpc
import internal_communication_pb2
import internal_communication_pb2_grpc

def get_telemetry():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = internal_communication_pb2_grpc.InternalServiceStub(channel)
        
        try:
            response = stub.GetTelemetry(internal_communication_pb2.Empty())
            
            print("Data:")
            print(f"Latitude: {response.current_latitude}")
            print(f"Longitude: {response.current_longitude}")
            
        except grpc.RpcError as e:
            print(f"Error: {e.code()}")

if __name__ == "__main__":
    get_telemetry()
