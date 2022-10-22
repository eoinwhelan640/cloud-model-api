import grpc
import protos.cats_vs_dogs_pb2 as cvd_pb2
import protos.cats_vs_dogs_pb2_grpc as cvd_pb2_grpc
from concurrent import futures
import logging
from models import Models


class CatsVsDogsService(cvd_pb2_grpc.CatsVsDogsServiceServicer):
    def CatsVsDogsTensorflowInference(self, request, context):
        try:
            logging.info("Running Tensorflow gRPC inference...")
            img_array = models.tf_load_image(request.image)
            result = models.tf_predict(img_array)
            return cvd_pb2.CatsVsDogsResponse(cls=result["class"])
        except Exception as e:
            message = "Server error while processing image!"
            logging.error(f"{message} {e}", exc_info=True)
            server_error = grpc.RpcError(message)
            server_error.code = lambda: grpc.StatusCode.INTERNAL
            raise server_error

    def CatsVsDogsPyTorchInference(self, request, context):
        try:
            logging.info("Running PyTorch gRPC inference...")
            img_tensor = models.pyt_load_image(request.image)
            result = models.pyt_predict(img_tensor)
            return cvd_pb2.CatsVsDogsResponse(cls=result["class"])
        except Exception as e:
            message = "Server error while processing image!"
            logging.error(f"{message} {e}", exc_info=True)
            server_error = grpc.RpcError(message)
            server_error.code = lambda: grpc.StatusCode.INTERNAL
            raise server_error

def serve():
    ''''
    Define gRPC. Declare what we're going to add to the server, ie what the endpoints are
    where we want it served at, ie secure or insecure connection by adding an in/secure port
    start it and wait for calls to come in
    '''
    server = grpc.server(futures.ThreadPoolExecutor())
    cvd_pb2_grpc.add_CatsVsDogsServiceServicer_to_server(CatsVsDogsService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    logging.info("Started gRPC server on localhost:50051...")
    # Server will run continously
    server.wait_for_termination()


if __name__ == "__main__":
    # Configure the logger
    logging.basicConfig(level=logging.INFO)
    models = Models()
    # Serve our api
    serve()

