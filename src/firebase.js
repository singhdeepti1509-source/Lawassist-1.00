import { initializeApp } from 'firebase/app';
import {
  getFirestore,
  collection,
  addDoc,
  query,
  where,
  getDocs,
  doc,
  setDoc,
  getDoc,
  serverTimestamp,
} from 'firebase/firestore';
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut,
} from 'firebase/auth';

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);
export const auth = getAuth(app);

console.log('Firebase initialized with project:', firebaseConfig.projectId);

// Save user data to Firestore
export const saveUserData = async (userId, userData) => {
  try {
    console.log('Saving user data for:', userId);
    await setDoc(
      doc(db, 'users', userId),
      {
        ...userData,
        createdAt: userData.createdAt || serverTimestamp(),
        updatedAt: serverTimestamp(),
      },
      { merge: true }
    );
    console.log('User data saved successfully to Firestore');
    return { success: true };
  } catch (error) {
    console.error('Error saving user data:', error);
    console.error('Error code:', error.code);
    console.error('Error message:', error.message);
    return { success: false, error };
  }
};

// Get user data from Firestore
export const getUserData = async (userId) => {
  try {
    console.log('Getting user data for:', userId);
    const docRef = doc(db, 'users', userId);
    const docSnap = await getDoc(docRef);

    if (docSnap.exists()) {
      console.log('User data found:', docSnap.data());
      return { success: true, data: docSnap.data() };
    } else {
      console.log('No user data found for:', userId);
      return { success: false, data: null };
    }
  } catch (error) {
    console.error('Error getting user data:', error);
    console.error('Error code:', error.code);
    console.error('Error message:', error.message);
    return { success: false, error, data: null };
  }
};

// Save chat log to Firestore
export const saveChatLog = async (userId, sessionId, userName, message) => {
  try {
    console.log('Saving chat log:', {
      userId,
      sessionId,
      userName,
      role: message.role,
      contentPreview: message.content.substring(0, 50) + '...'
    });

    const chatData = {
      userId,
      sessionId,
      userName,
      role: message.role,
      content: message.content,
      timestamp: serverTimestamp(),
      createdAt: new Date().toISOString(),
    };

    const docRef = await addDoc(collection(db, 'chat_logs'), chatData);
    console.log('Chat log saved successfully with ID:', docRef.id);
    return { success: true, id: docRef.id };
  } catch (error) {
    console.error('Error saving chat log:', error);
    console.error('Error code:', error.code);
    console.error('Error message:', error.message);
    console.error('Error details:', error);
    return { success: false, error };
  }
};

// Get chat history from Firestore
export const getChatHistory = async (userId) => {
  try {
    console.log('Fetching chat history for user:', userId);
    
    // Query without orderBy to avoid index issues
    const q = query(
      collection(db, 'chat_logs'),
      where('userId', '==', userId)
    );
    
    const querySnapshot = await getDocs(q);
    const logs = [];
    
    querySnapshot.forEach((docSnap) => {
      const data = docSnap.data();
      logs.push({
        id: docSnap.id,
        ...data,
        // Normalize timestamp
        timestamp:
          data.createdAt ||
          (data.timestamp && data.timestamp.toDate
            ? data.timestamp.toDate().toISOString()
            : new Date().toISOString()),
      });
    });
    
    // Sort in memory instead of using Firestore orderBy
    logs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    
    console.log(`Loaded ${logs.length} chat logs for user ${userId}`);
    return { success: true, logs };
  } catch (error) {
    console.error('Error loading chat history:', error);
    console.error('Error code:', error.code);
    console.error('Error message:', error.message);
    return { success: false, logs: [], error };
  }
};

export { createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut };