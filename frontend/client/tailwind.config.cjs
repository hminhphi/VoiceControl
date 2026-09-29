/** @type {import('tailwindcss').Config} */
export default {
    // 1. Tell Tailwind which files contain your class names (for purging/tree-shaking)
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {},
    },
    plugins: [],
}