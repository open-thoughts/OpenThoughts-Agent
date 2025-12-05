## Run SkyRL training with proxies

Full example script is [here](train_skyrl.sh).

In order to be able to connect to daytona from compute nodes, use [proxychains](https://github.com/rofl0r/proxychains-ng).

To install:

```bash
git clone https://github.com/rofl0r/proxychains-ng.git
cd proxychains-ng
./configure --prefix=$HOME/.local
make
make install
export PATH=$HOME/.local/bin:$PATH
```

This will create a shared object `~/.proxychains/proxychains.conf`.

Setup SHH tunnel (pay attention to `-g` it allows other nodes to see this tunnel)
```bash
 ssh -g -f -N -D ${head_node_ip}:$PORT_TO_USE \
                    -o StrictHostKeyChecking=no \
                    -o ConnectTimeout=1000 \
                    -o ServerAliveInterval=15 \
                    -o ServerAliveCountMax=15 \
                    -o TCPKeepAlive=no \
                    -o ExitOnForwardFailure=yes \
                    -o BatchMode=yes \
                    -i $SSH_KEY \
                    ${USER_NAME}@$LOGIN_NODE && \
```

Configure proxychains (should be saved somewhere like  `~/.proxychains/proxychains.conf`):
```
strict_chain
proxy_dns
tcp_read_time_out  30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
localnet 127.0.0.1/255.255.255.255
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
[ProxyList]
socks5  10.14.0.145 25001 <- your head node IP and the port of your proxy
```

Setting environment variables (set up AFTER establishing the SSH tunnel):
```bash
LD_PRELOAD=~/.local/lib/libproxychains4.so 
PROXYCHAINS_CONF=~/.proxychains/proxychains.conf
```

