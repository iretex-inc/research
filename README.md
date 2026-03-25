# SHAP Analysis Studio

Upload any CSV/Excel dataset → select a target variable → get 7 publication-quality SHAP plots, each exportable as SVG.

---

## Quick Start

```bash
# 1. Clone / copy this folder onto your server
scp -r shap-studio/ user@your-server:~/

# 2. SSH in
ssh user@your-server
cd shap-studio

# 3. Configure port (optional — default is 8080)
cp .env.example .env
nano .env          # change PORT= if needed

# 4. Build and start
docker compose up -d --build

# 5. Open in browser
http://your-server-ip:8080
```

---

## Commands

| Task | Command |
|------|---------|
| Start | `docker compose up -d` |
| Stop | `docker compose down` |
| Rebuild after update | `docker compose up -d --build` |
| View logs | `docker compose logs -f` |
| Check health | `docker compose ps` |
| Shell into container | `docker exec -it shap-studio sh` |

---

## Updating the App

```bash
# Pull new version (if using git) or copy new files
git pull   # or scp new files

# Rebuild and restart (zero-downtime approach)
docker compose up -d --build
```

---

## Reverse Proxy (Optional)

### Nginx on host

Add to your `/etc/nginx/sites-available/shap`:

```nginx
server {
    listen 80;
    server_name shap.yourdomain.local;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/shap /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Traefik

Uncomment the Traefik labels in `docker-compose.yml` and set your domain.

---

## Architecture

```
Browser
  └── Port 8080 (configurable)
        └── Docker container: nginx:1.27-alpine
              └── /usr/share/nginx/html  ← Vite production build
                    └── React SPA (single bundle, ~150 KB gzipped)
```

**Build process (multi-stage Docker):**
1. `node:20-alpine` — installs deps, runs `vite build`
2. `nginx:1.27-alpine` — copies `dist/`, serves with gzip + caching headers

**Runtime:** Pure static files. No backend, no database, no API keys.  
All computation (OLS, SHAP) runs **in the browser**. Your data never leaves your machine.

---

## Requirements

- Docker 24+ and Docker Compose v2
- ~200 MB disk (image is ~25 MB compressed)
- Any CPU, 32 MB RAM minimum

Tested on: Ubuntu 22.04, Debian 12, Raspberry Pi OS (arm64)

---

## Plots Generated

| # | Name | Description |
|---|------|-------------|
| 1 | Feature Importance Bar | Mean \|SHAP\| ranked, direction-coloured |
| 2 | Beeswarm Summary | All observations, viridis feature-value colour |
| 3 | Dependence Plots | Top-4: feature value vs SHAP + trend line |
| 4 | Waterfall | Low / median / high prediction decomposition |
| 5 | Heatmap | Full SHAP matrix sorted by predicted value |
| 6 | Force Plot | Stacked contributions, all observations |
| 7 | Summary Table | Coefficients, R², RMSE, ranked features |

Every plot exports as **SVG vector** (journal-ready, infinitely scalable).

---

## License

MIT — use freely, modify as needed.
