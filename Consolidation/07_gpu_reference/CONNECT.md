# GPU laptop — connection recipe

Host: `vishwa@100.116.22.29` (Tailscale).
**Password: see `../03_round3_working_repo/AGENTS.md` §5.** It is deliberately not duplicated
here, so this file stays safe to share if `Consolidation/` is ever opened up.

The Mac has **no `sshpass` and no `timeout`**. Use `SSH_ASKPASS`:

```bash
cat > /tmp/askpass.sh <<'EOF'
#!/bin/sh
echo "$GPU_PASSWORD"
EOF
chmod +x /tmp/askpass.sh
export GPU_PASSWORD='<see 03_round3_working_repo/AGENTS.md §5>'

SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
      -o PubkeyAuthentication=no vishwa@100.116.22.29 '<command>'
```

The same env prefix works for `scp`:

```bash
SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  scp -o StrictHostKeyChecking=no -r vishwa@100.116.22.29:'~/Desktop/r3_runtime/Phase_6/results/' ./
```

## Standing rules

1. **Read-only**, except `/tmp` and `~/Desktop/r3_runtime/Phase_6/`. Never create, edit or delete
   anything under `~/Desktop/AISEHack-2.0/`.
2. **One heavy job at a time.** Others use that machine.
3. Long jobs: `nohup … > log 2>&1 &`, then poll the log. Do not hold an SSH session open for
   hours.
4. Copy **all** results back into this repository; the laptop is runtime, not storage.
5. Clean up any `/tmp` scratch directory you create.
6. **Never write the password into a repository file.**
