import grpc
import protos.cats_vs_dogs_pb2 as cvd_pb2
import protos.cats_vs_dogs_pb2_grpc as cvd_pb2_grpc

# Video gRPC 4
graham_img = "../images/graham2.jpg"
channel = grpc.insecure_channel("localhost:50051")
client = cvd_pb2_grpc.CatsVsDogsServiceStub(channel)

with open(graham_img, "rb") as img_file:
    img_bytes = img_file.read()


request = cvd_pb2.CatsVsDogsRequest(image=img_bytes)
tf_response = client.CatsVsDogsTensorflowInference(request)
pyt_response = client.CatsVsDogsPyTorchInference(request)

print("tf: ", tf_response, "pytorch: ", pyt_response)
