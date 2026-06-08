# Contributing to Delhi Election Dashboard

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Election-Dashboard.git
   cd Election-Dashboard
   ```
3. **Set up the development environment** following [SETUP_GUIDE.md](./SETUP_GUIDE.md)
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Frontend Development

The frontend uses React + Vite:

```bash
npm run dev
```

- Source files: `src/`
- Components: `src/components/`
- Utilities: `src/shared/`
- Styling: Tailwind CSS (edit `tailwind.config.js`)

### Backend Development

The backend uses Flask:

```bash
python model_api.py
```

- Main API: `model_api.py`
- ML Logic: `app1.py`
- Helper scripts: `check_*.py`, `analyze_*.py`

### Code Style

**JavaScript/React:**
- Use functional components with hooks
- Follow ESLint configuration
- Use meaningful variable names
- Add comments for complex logic

**Python:**
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions
- Keep functions focused and small

## Making Changes

### Adding New Features

1. **Update documentation** if adding new endpoints or features
2. **Test thoroughly** before submitting
3. **Keep changes focused** - one feature per PR
4. **Add comments** for complex logic

### Fixing Bugs

1. **Describe the bug** in your PR
2. **Explain the fix** and why it works
3. **Test the fix** thoroughly
4. **Check for side effects**

## Pull Request Process

1. **Update README.md** if needed
2. **Test your changes**:
   ```bash
   # Frontend
   npm run build  # Ensure builds without errors
   
   # Backend
   python model_api.py  # Ensure starts without errors
   ```
3. **Commit with clear messages**:
   ```bash
   git commit -m "Add: New booth visualization feature"
   git commit -m "Fix: Map not loading on mobile devices"
   ```
4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Create Pull Request** on GitHub

## Types of Contributions

### Code Contributions
- New features
- Bug fixes
- Performance improvements
- Code refactoring

### Documentation
- Improving README
- Adding code comments
- Creating tutorials
- Fixing typos

### Testing
- Writing tests
- Reporting bugs
- Testing on different platforms

### Design
- UI/UX improvements
- Creating icons/graphics
- Improving accessibility

## Project Structure Understanding

```
Election-Dashboard/
├── src/               # React frontend
│   ├── components/    # UI components
│   ├── shared/        # Utilities and helpers
│   └── App.jsx        # Main app
├── public/data/       # Static data files
├── model_api.py       # Flask API server
├── app1.py            # ML prediction logic
└── requirements.txt   # Python dependencies
```

## Data Handling Guidelines

⚠️ **IMPORTANT**: Never commit sensitive voter data!

- **DO NOT** commit `.xlsx` Excel files
- **DO NOT** commit personal voter information
- **DO** use sample/anonymized data for testing
- **DO** respect `.gitignore` rules

Files to never commit:
- `*.xlsx`, `*.xls` - Excel voter data
- `*.pkl`, `*.pth` - Large model files
- `.env` - Environment configuration
- Personal API keys or credentials

## Testing Guidelines

Before submitting:

1. **Frontend tests**:
   ```bash
   npm run build  # Should complete without errors
   ```

2. **Backend tests**:
   ```bash
   python model_api.py  # Should start successfully
   ```

3. **Manual testing**:
   - Test all 4 dashboard levels (Parliament → Assembly → Ward → Booth)
   - Test map interactions
   - Test charts and visualizations
   - Test on different screen sizes

## Getting Help

- **Questions**: Open a GitHub issue with the `question` label
- **Bugs**: Open a GitHub issue with the `bug` label
- **Features**: Open a GitHub issue with the `enhancement` label

## Code Review Process

- All PRs will be reviewed by maintainers
- Feedback will be provided for improvements
- Changes may be requested before merging
- Be patient and respectful in discussions

## Recognition

Contributors will be:
- Listed in the project README
- Credited in release notes
- Appreciated in the community!

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to the Delhi Election Dashboard! 🎉
