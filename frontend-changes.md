# Frontend Changes - Dark/Light Theme Toggle

## Overview
Implemented a dark/light theme toggle feature for the RAG chatbot application with smooth transitions and full accessibility support.

## Changes Made

### 1. HTML (index.html)
- Added theme toggle button in the sidebar header with sun/moon SVG icons
- Button positioned next to the "New Chat" button using flexbox layout
- Includes proper ARIA labels and keyboard accessibility attributes

### 2. CSS (style.css)
- **Theme Variables**: Created comprehensive CSS variable system for both dark and light themes
  - Dark theme (default): Dark blue/gray color scheme for comfortable night viewing
  - Light theme: Clean white/light gray scheme with good contrast
- **Smooth Transitions**: Added 0.3s ease transitions to all theme-affected elements
- **Theme Toggle Button**: Circular button with hover rotation effect and icon transitions
- **Icon Animations**: Sun/moon icons smoothly fade and rotate when switching themes
- **Component Updates**: Updated all UI components to use theme variables:
  - Background colors
  - Text colors  
  - Border colors
  - Surface colors
  - Code block styling
  - Input fields and buttons

### 3. JavaScript (script.js)
- **Theme Initialization**: Loads saved theme preference from localStorage on page load
- **Toggle Functionality**: Switches between dark/light themes with single click
- **Persistence**: Saves theme preference to localStorage for future visits
- **Keyboard Support**: Full keyboard navigation (Enter/Space keys)
- **Accessibility**: 
  - Dynamic ARIA labels that update based on current theme
  - Screen reader announcements for theme changes
  - Proper focus management

## Features
- **Icon-based Design**: Clean sun/moon icons that clearly indicate current theme
- **Smooth Animations**: 
  - Button rotates 180° on hover
  - Icons fade and scale during transition
  - All color changes animate smoothly
- **Fully Accessible**:
  - Keyboard navigable (Tab to focus, Enter/Space to activate)
  - Screen reader compatible with announcements
  - Proper ARIA labels
- **Persistent**: Theme choice saved across sessions
- **Responsive**: Works on all screen sizes

## Technical Implementation
- Uses `data-theme` attribute on document root for theme switching
- CSS variables cascade throughout the application
- No external dependencies - uses vanilla JavaScript
- Lightweight implementation with minimal performance impact