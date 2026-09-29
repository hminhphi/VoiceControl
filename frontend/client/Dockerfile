# frontend/client/Dockerfile

FROM node:20-alpine

WORKDIR /app

# Install dependencies first with package.json and package-lock.json (for better cache)
COPY package*.json ./

RUN rm -rf node_modules package-lock.json
RUN npm install

# Copy rest of the app
COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]