<think>
The user wants a complete, production-grade distributed Actor Framework in C++20. Let me break down what's needed:

1. **Lock-free MPSC mailbox** - Multiple producers, single consumer. Uses atomic intrusive linked list (CAS-based push, pop from head).

2. **Actor base class** - Template with lifecycle hooks: `pre_start()`, `post_stop()`, `receive(msg)`.

3. **ActorSystem** - Thread pool with work-stealing (each thread has a deque, steal from random other thread), affinity masks (pthread affinity / std::thread::hardware_concurrency).

4. **Typed message passing** - `std::variant` for message types, `std::visit` for dispatch.

5. **Supervisor tree** - OneForOne and AllForOne strategies, max restarts per time window.

6. **Zero-copy binary serialization** - Envelope with header (magic, version, msg type, size, checksum) + payload. Zero-copy means we serialize into a buffer and send via sendmsg/recvmsg with iovec.

Let me write this as a single comprehensive header + implementation. I'll structure it as:

- `actor_framework.hpp` - main header
- Or just one big file since they said "full, functional, clean C++20 code"

Let me think about the structure:

```
// 1. MPSC Mailbox
template<typename T>
class Mailbox {
  // intrusive linked list
  struct Node { T value; Node* next; };
  Node* head; // CAS for push
  Node* tail; // for pop
  // push: CAS on head
  // pop: swap head
};

// 2. Actor base
template<typename... Msgs>
class Actor {
  virtual void pre_start();
  virtual void post_stop();
  virtual void receive(std::variant<Msgs...>&& msg);
};

// 3. ActorSystem with work-stealing pool
class ActorSystem {
  // thread pool with per-thread deque
  // work stealing: pop from own, then steal from random
  // affinity: set on each thread
};

// 4. Typed messages
// std::variant<Msg1, Msg2, ...>
// visitor pattern in receive

// 5. Supervisor
enum class Strategy { OneForOne, AllForOne };
class Supervisor {
  // max restarts per window
};

// 6. Binary envelope
struct Envelope {
  uint32_t magic;
  uint32_t version;
  uint32_t msg_type;
  uint32_t size;
  uint32_t checksum;
  // payload follows
};
```

Let me write the full implementation now. I'll make it a single header file for clarity.

Key design decisions:
- Mailbox: CAS-based push (MPSC), swap-based pop (MCS). Use `std::atomic<Node*>` for head.
- Actor: CRTP or virtual. I'll use virtual for simplicity with variant.
- Thread pool: Each worker has a `std::deque` (or vector) of actor pointers. Work stealing: pop from own deque, then steal from random other.
- Messages: `std::variant` with `std::visit`.
- Supervisor: Tree with parent/child, strategy enum, restart counter with sliding window.
- Serialization: Struct with packed layout, CRC32 checksum, iovec for send/recv.

Let me write it all out properly.
</think>

# Distributed Actor Framework — C++20

```cpp
// ============================================================================
// actor_framework.hpp
// A production-grade distributed Actor framework in C++20.
//
// Features:
//   1. Lock-free MPSC mailbox (atomic intrusive linked list)
//   2. Actor base with pre_start / post_stop / receive
//   3. ActorSystem with work-stealing thread pool + CPU affinity
//   4. Typed messages via std::variant + std::visit
//   5. Supervisor tree (OneForOne / AllForOne) with rate-limited restarts
//   6. Zero-copy binary envelope (CRC-32, iovec send/recv)
//
// Build: g++ -std=c++20 -O2 -pthread -o app main.cpp
// ============================================================================

#pragma once

#include <atomic>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>
#include <algorithm>
#include <bit>
#include <concepts>
#include <format>
#include <print>

// POSIX for sockets + affinity
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/uio.h>

namespace actor {

// ============================================================================
// Forward declarations
// ============================================================================
class ActorSystem;
class Actor;

// ============================================================================
// 1. Lock-free MPSC Mailbox
// ============================================================================

template <typename T>
class Mailbox {
public:
    struct Node {
        T value;
        std::atomic<Node*> next;
        explicit Node(T&& v) : value(std::move(v)) { next.store(nullptr, std::memory_order_relaxed); }
    };

    Mailbox() = default;
    ~Mailbox() {
        Node* n = head_.load(std::memory_order_relaxed);
        while (n) {
            Node* next = n->next.load(std::memory_order_relaxed);
            delete n;
            n = next;
        }
    }

    Mailbox(const Mailbox&) = delete;
    Mailbox& operator=(const Mailbox&) = delete;

    // Multiple producers: CAS on head_
    bool push(T&& value) {
        Node* node = new Node(std::move(value));
        Node* old_head;
        do {
            old_head = head_.load(std::memory_order_acquire);
            node->next.store(old_head, std::memory_order_relaxed);
        } while (!head_.compare_exchange_weak(
            old_head, node,
            std::memory_order_release,
            std::memory_order_relaxed));
        return true;
    }

    // Single consumer: swap head_
    bool pop(T& out) {
        Node* old_head = head_.exchange(nullptr, std::memory_order_acquire);
        if (!old_head) return false;
        out = std::move(old_head->value);
        delete old_head;
        return true;
    }

    // Pop all (for drain)
    std::vector<T> drain() {
        std::vector<T> result;
        T val;
        while (pop(val)) {
            result.push_back(std::move(val));
        }
        return result;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) == nullptr;
    }

    size_t size_approx() const {
        size_t n = 0;
        for (Node* p = head_.load(std::memory_order_acquire); p;
             p = p->next.load(std::memory_order_relaxed))
            ++n;
        return n;
    }

private:
    std::atomic<Node*> head_{nullptr};
};

// ============================================================================
// 2. Actor base class
// ============================================================================

enum class ActorState {
    New, Starting, Running, Stopping, Stopped, Restarting
};

enum class ActorFailure {
    None,
    Restart,
    Stop,
    Escalate
};

struct ActorRef {
    Actor* actor;
    ActorRef() : actor(nullptr) {}
    explicit ActorRef(Actor* a) : actor(a) {}
};

class Actor {
public:
    virtual ~Actor() = default;

    // Lifecycle hooks
    virtual void pre_start() {}
    virtual void post_stop() {}

    // Message handler — override in derived
    virtual void receive(std::variant<>&& msg) = 0;

    // Control messages
    struct Start { int id; };
    struct Stop { int id; };
    struct Restart { int id; };
    struct Poison { int id; };

    using ControlMsg = std::variant<Start, Stop, Restart, Poison>;

    // Send a message to another actor
    void send(ActorRef target, std::variant<>&& msg) {
        target.actor->mailbox_.push(std::move(msg));
    }

    // Reply to sender
    void reply(std::variant<>&& msg) {
        if (sender_) send(sender_, std::move(msg));
    }

    // Self reference
    ActorRef self() { return {this}; }

    // Ask pattern
    template <typename Reply>
    void ask(ActorRef target, std::variant<>&& msg) {
        send(target, std::move(msg));
    }

