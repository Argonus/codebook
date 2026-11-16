FROM node:24-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev
COPY author.json ./
COPY frontend ./frontend
COPY scripts ./scripts
COPY server ./server
COPY tests ./tests
RUN node scripts/build.mjs
USER node
ENV NODE_ENV=production
EXPOSE 8080
CMD ["node", "server/main.mjs"]
