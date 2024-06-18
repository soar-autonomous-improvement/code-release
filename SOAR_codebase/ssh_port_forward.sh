ssh -L 6000:localhost:5000 -N -f -C -o ExitOnForwardFailure=yes $USER@<ip address of server>
ssh -L 7000:localhost:6000 -N -f -C -o ExitOnForwardFailure=yes $USER@<ip address of server>
