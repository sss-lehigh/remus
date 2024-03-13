# Documentation for TCP module

## What are the classes

message - A struct for communication. Can contain 4 "8-byte" numbers.

SocketManager - A class for acting as a server and managing sockets with a list of clients. The socket manager can only send a blast message to all the clients.

EndpointManager - A class for acting as a client. Since the socket manager can only send and receive from all the clients at once, these endpoints function like a "herd".

N.B. If one endpoint manager stopped communication while another endpoint manager continued to communicate with the server, then the socket manager blocks waiting for a message from the retired instance of the endpoint. (The socket manager cannot retire the connection of a singular endpoint)

## Example

```{c++}
#include <thread>
#include <vector>
#include <remus/util/tcp/tcp.h>
#include <remus/rdma/memory_pool/memory_pool.h>
#include <remus/logging/logging.h>

#define PORT_NUM 18000
#define CLIENT_COUNT 4

using remus::rdma::MemoryPool;

int main(){
    ROME_INIT_LOG();
    MemoryPool::Peer host = MemoryPool::Peer(0, "localhost", PORT_NUM - 1);
    std::vector<std::thread> threads;
    // Create a socket manager for 4 connections
    threads.emplace_back(std::thread([&](){
        // Create a socket manager
        tcp::SocketManager socket_handle = tcp::SocketManager(PORT_NUM);
        for(int i = 0; i < CLIENT_COUNT; i++){
            // Accept 4 connections to that socket
            socket_handle.accept_conn();
        }
        // Recv a message from every client
        tcp::message recv_messages[CLIENT_COUNT];
        socket_handle.recv_from_all(recv_messages);
        ROME_INFO("Received from clients");
        // Once this happens, all clients have synced, so we can broadcast

        tcp::message ptr_message = tcp::message(30); 
        // Instead of sending 30, you could imagine a use case of broadcasting a remote_ptr
        socket_handle.send_to_all(&ptr_message);
        ROME_INFO("Sent to clients");
    }));

    // While the server is waiting for connections, we need to spin up clients
    for(int i = 0; i < CLIENT_COUNT; i++){
        threads.emplace_back(std::thread([&](int index){
            // Create an endpoint
            tcp::EndpointManager endpoint = tcp::EndpointManager(PORT_NUM, host.address.c_str());

            // Send the server a message
            tcp::message ptr_message = tcp::message(10); 
            endpoint.send_server(&ptr_message);
            ROME_INFO("Synced client {} with server", index);
            // Sync between the server and endpoints
            endpoint.recv_server(&ptr_message);
            ROME_INFO("Client {} received server message", index);
            // ptr_message should be the server's value now...
            assert(ptr_message.get_first() == 30);
        }, i));
    }

    for (auto it = threads.begin(); it != threads.end(); it++){
        it->join();
    }
    ROME_INFO("All threads done");
    return 0;
}
```

Output:
> [2023-10-07 04:16:08.557] [info] [test.cc:40] Syncing client 0 with server<br>
> [2023-10-07 04:16:08.557] [info] [test.cc:40] Syncing client 1 with server<br>
> [2023-10-07 04:16:08.557] [info] [test.cc:40] Syncing client 3 with server<br>
> [2023-10-07 04:16:08.558] [info] [test.cc:40] Syncing client 2 with server<br>
> [2023-10-07 04:16:08.558] [info] [test.cc:26] Received from clients<br>
> [2023-10-07 04:16:08.559] [info] [test.cc:31] Sending to clients<br>
> [2023-10-07 04:16:08.559] [info] [test.cc:44] Client 0 received server message<br>
> [2023-10-07 04:16:08.559] [info] [test.cc:44] Client 1 received server message<br>
> [2023-10-07 04:16:08.559] [info] [test.cc:44] Client 3 received server message<br>
> [2023-10-07 04:16:08.559] [info] [test.cc:44] Client 2 received server message<br>
> [2023-10-07 04:16:08.560] [info] [test.cc:53] All threads done<br>
