/**
 * @author Ethan Lavi
 * @brief A simple communication primitive for exchanging information within a
 * network. Utilizes a server-client approach to broadcast entire messages
 */
// ! I think this file should be refactored into a RDMA barrier and some other
// internode message-passing that doesn't rely on protobufs. ! For now, it
// serves as a temporary and cleaner API for these things while they are in
// development

#pragma once

#include "remus/logging/logging.h"
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <netdb.h>
#include <netinet/in.h>
#include <stdexcept>
#include <stdlib.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

/// The size of each message, sent over the EndpointManager and the
/// SocketManager
#define TCP_MESSAGE_SIZE 32

/// TCP library
namespace remus::util::tcp {
/// @brief A struct as a standard for communicating as a server-client. It has
/// the capability to store 4 64-bit ints. In practice, I've only needed one,
/// but the extra space could come in handy.
struct message {
  union {
    struct {
      uint64_t first;
      uint64_t second;
      uint64_t third;
      uint64_t fourth;
    } ints;
    char data[TCP_MESSAGE_SIZE];
  } content;
  char padding = '\0'; // a null termination byte, in case I want to create a
                       // null-terminated char*

  /// @brief Construct a message.
  /// @param first_ pack the first part of the message, defaults to 0
  /// @param second_ pack the second part of the message, defaults to 0
  /// @param third_ pack the third part of the message, defaults to 0
  /// @param fourth_ pack the fourth part of the message, defaults to 0
  message(uint64_t first_ = 0, uint64_t second_ = 0, uint64_t third_ = 0,
          uint64_t fourth_ = 0) {
    content.ints.first = first_;
    content.ints.second = second_;
    content.ints.third = third_;
    content.ints.fourth = fourth_;
  }

  /// @brief unpack the first part of the message, default 0
  /// @return get the first value
  uint64_t get_first() { return content.ints.first; }
  /// @brief unpack the second part of the message, default 0
  /// @return get the second value
  uint64_t get_second() { return content.ints.second; }
  /// @brief unpack the third part of the message, default 0
  /// @return get the third value
  uint64_t get_third() { return content.ints.third; }
  /// @brief unpack the fourth part of the message, default 0
  /// @return get the fourth value
  uint64_t get_fourth() { return content.ints.fourth; }
};

/// @brief A class for acting as a server and managing sockets with a list of
/// clients. The interface is the server can only communicate with all the
/// clients at once (sort of like as a herd)
/// @attention The socket manager cannot retire the singular connections of an
/// endpoint. As a consequence of this, if one endpointmanager stopped
/// communication while another endpointmanager continued to communicate with
/// the server, then the socket manager blocks waiting for a message from the
/// retired instance of the endpoint.
class SocketManager {
private:
  /// @brief Throw an exception with an error message
  /// @param message the message to print
  void error(const char *message) {
    REMUS_WARN(strerror(errno));
    throw std::runtime_error(message);
  }

  // socket for accepting new connections
  int sockfd = -1;
  // socket information
  struct sockaddr_in address;
  // Client sockets
  std::vector<int> client_sockets;
  // If this object owns the socket
  bool owner = true;

  /// @brief Release the object of socket closing responsibility
  void release() { owner = false; }

public:
  // Define copy/move for endpoint manager
  SocketManager(SocketManager &obj) {
    sockfd = obj.sockfd;
    client_sockets = obj.client_sockets;
    address = obj.address;
    obj.release();
  }
  SocketManager &operator=(SocketManager &obj) {
    sockfd = obj.sockfd;
    client_sockets = obj.client_sockets;
    address = obj.address;
    obj.release();
    return *this;
  }
  SocketManager(SocketManager &&obj) {
    sockfd = obj.sockfd;
    client_sockets = obj.client_sockets;
    address = obj.address;
    obj.release();
  }
  SocketManager &operator=(SocketManager &&obj) {
    sockfd = obj.sockfd;
    client_sockets = obj.client_sockets;
    address = obj.address;
    obj.release();
    return *this;
  }

  /// @brief Accept a new client into our list of clients
  bool accept_conn() {
    // Accept new connection
    int address_size = sizeof(address);
    int client_sockfd =
        accept(sockfd, (struct sockaddr *)&address, (socklen_t *)&address_size);
    if (client_sockfd == -1)
      return false;
    // Record it
    client_sockets.push_back(client_sockfd);
    return true;
  }

  /// @brief Send a message to every client
  /// @param send_buffer the message to send to each client
  void send_to_all(message *send_buffer) {
    for (int i = 0; i < client_sockets.size(); i++) {
      write(client_sockets[i], send_buffer->content.data, TCP_MESSAGE_SIZE + 1);
    }
  }

  /// @brief The number of clients managed by the server
  int num_clients() { return client_sockets.size(); }

  /// @brief Receive a message from all the clients. Make sure recv_buffer has
  /// enough room for all num_clients()
  /// @param recv_buffer an array of message of size num_clients(). This
  /// function will modify this buffer.
  void recv_from_all(message *recv_buffer) {
    for (int i = 0; i < client_sockets.size(); i++) {
      read(client_sockets[i], recv_buffer[i].content.data,
           TCP_MESSAGE_SIZE + 1);
    }
  }

  // Default constructor
  SocketManager() { owner = false; }

  /// @brief Create an instance of the socket manager. Will throw an exception
  /// if cannot connect to the socket. (Not thread-safe)
  /// @param port The port to host/connect to this SocketManager on.
  /// @param expected_conn The backlog to create when listening on the socket.
  /// If this is too low, then too many requests to connect to this socket could
  /// fail.
  SocketManager(uint16_t port, int expected_conn = 10) {
    // Create a new socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    int result = 0;
    if (sockfd == -1)
      error("Cannot open socket");
    // Make sure we can re-use the socket immediately after deleting
    const int enable = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) <
        0) {
      error("setsockopt failed");
    }
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(int)) <
        0) {
      error("setsockopt failed");
    }

    // Bind it
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = INADDR_ANY;
    result = bind(sockfd, (struct sockaddr *)&address, sizeof(address));
    if (result == -1) {
      close(sockfd);
      error("Cannot bind to socket");
    }
    // Listen for expected_conn
    result = listen(sockfd, expected_conn);
    if (result == -1) {
      close(sockfd);
      error("Cannot listen on socket");
    }
  }

  /// @brief Destroy all the resources
  ~SocketManager() {
    // If we aren't the owner, it isn't our job to close the sockets
    if (!owner)
      return;
    // Close each client socket
    for (int i = 0; i < client_sockets.size(); i++) {
      close(client_sockets[i]);
    }
    if (sockfd != -1) {
      // Only close the main socket if safe to do so (sockfd can be -1)
      close(sockfd);
    }
  }
};

/// @brief A class for acting as a client to connect to SocketManager. This
/// class must be used with a herd-mentality --> anything one client does, all
/// should do
/// @attention The socket manager cannot retire the singular connections of an
/// endpoint. As a consequence of this, if one instance stopped communication
/// while another instance continued to communicate with the server, then the
/// socket manager blocks waiting for a message from the retired instance of the
/// endpoint.
class EndpointManager {
private:
  /// @brief Throw an exception with an error message
  /// @param message the message to print
  void error(const char *message) {
    REMUS_WARN(strerror(errno));
    throw std::runtime_error(message);
  }

  // Socket for communication with the server
  int sockfd = -1;
  bool owner = true;

  // Release responsibility of closing the socket
  void release() { owner = false; }

public:
  // Define copy/move for endpoint manager
  EndpointManager(EndpointManager &obj) {
    sockfd = obj.sockfd;
    obj.release();
  }
  EndpointManager &operator=(EndpointManager &obj) {
    sockfd = obj.sockfd;
    obj.release();
    return *this;
  }
  EndpointManager(EndpointManager &&obj) {
    sockfd = obj.sockfd;
    obj.release();
  }
  EndpointManager &operator=(EndpointManager &&obj) {
    sockfd = obj.sockfd;
    obj.release();
    return *this;
  }

  /// @brief Send a message to the server
  /// @param send_buffer the message to send to the server
  void send_server(message *send_buffer) {
    int status = write(sockfd, send_buffer->content.data, TCP_MESSAGE_SIZE + 1);
    if (status == -1)
      error("Cannot send data over socket");
  }

  /// @brief receive a message from the server
  /// @param recv_buffer a single message to receive into
  void recv_server(message *recv_buffer) {
    int status = read(sockfd, recv_buffer->content.data, TCP_MESSAGE_SIZE + 1);
    if (status == -1)
      error("Cannot read data over socket");
  }

  // Default constructor
  EndpointManager() { owner = false; }

  /// @brief Create an endpoint. Will connect to host by resolving hostname (for
  /// example, on cloudlab, using nodeX sufficies to identify the intended node)
  /// @param port The port to connect to the SocketManager on
  /// @param hostname The hostname the SocketManager is on
  EndpointManager(uint16_t port, const char *hostname) {
    // create a socket
    int result = 0;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1)
      error("Cannot open socket");
    // Get host by name (i.e. node0)
    struct addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_INET;
    if (getaddrinfo(hostname, NULL, &hints, &res) != 0) {
      close(sockfd);
      error("Cannot resolve hostname");
    }
    // Resolve IP from response
    struct in_addr host_ip = ((struct sockaddr_in *)(res->ai_addr))->sin_addr;
    freeaddrinfo(res);

    // Load socket information
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    // use the host ip for communication
    result = inet_pton(AF_INET, inet_ntoa(host_ip),
                       &address.sin_addr); // Pointer (to String) to Number.
    if (result <= 0) {
      close(sockfd);
      error("Cannot inet pton");
    }
    // Connect to socket
    result = connect(sockfd, (struct sockaddr *)&address, sizeof(address));
    while (result != 0) {
      // will spin as many times as necessary until can connect to server
      std::this_thread::sleep_for(std::chrono::milliseconds(250));
      result = connect(sockfd, (struct sockaddr *)&address, sizeof(address));
    }
  }

  /// @brief release the resources
  ~EndpointManager() {
    if (sockfd != -1 && owner) {
      close(sockfd);
    }
  }
};
// end namespace
} // namespace tcp
