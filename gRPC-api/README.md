# gRPC

We're starting with a proto file, which describes exactly what the gRPC api is going to do. It'll
strictly require the caller to have a certain payload. The caller needs to know what to give and
what to expect back format wise. We compile these files and can import them into our API so that
everything is strictly typed.

Generate the pb2 files via the grpcio & grpc-tools libraries. Run the below command from the gRPC folder.   
`python -m grpc_tools.protoc --python_out=protos --grpc_python_out=protos cats_vs_dogs.proto --proto_path=protos`

It spawns all the relevant api setup for us. 


#### REMEMBER TO uninstall protobuf 3.20.1 for the latest version. The _pb2 files were generated using that lib but tensorflow needs later/est version to work