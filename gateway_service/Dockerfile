FROM node:20-alpine

WORKDIR /app

COPY package.json tsconfig.json ./
RUN npm install

COPY src ./src
COPY public ./public
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
