import { dirname } from 'path';
import { fileURLToPath } from 'url';
import { FlatCompat } from '@eslint/eslintrc';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const prettierPlugin = await import('eslint-plugin-prettier');

const eslintConfig = [
  ...compat.extends('next/core-web-vitals', 'next/typescript', 'prettier'),
  {
    ignores: [
      '**/.next/**',
      '**/node_modules/**',
      '**/build/**',
      '**/dist/**',
      '**/coverage/**',
      '**/out/**',
      '**/.turbo/**',
    ],
  },
  {
    plugins: {
      prettier: prettierPlugin.default,
    },
    rules: {
      'prettier/prettier': 'error',
    },
  },
];

export default eslintConfig;
